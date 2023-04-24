#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import checkpoint_sequential


class MultiheadAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, size, device):
        if not hasattr(cls, '_future_mask') or cls._future_mask.device != device or cls._future_mask.shape < size:
            cls._future_mask = torch.triu(torch.ones(size[0], size[1], dtype=torch.bool, device=device), 1)

        mask = cls._future_mask[:size[0], :size[1]]

        return mask

    def __init__(self, n_features, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _split_heads(self, x, is_key=False):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v, apply_future_mask=True, padding_mask=None):
        # Q*K
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)

        if apply_future_mask:
            future_mask = MultiheadAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0)
            w.masked_fill_(future_mask, float('-inf'))
        
        if padding_mask is not None: # MASK
            w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Softmax(Q*K)
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        if padding_mask is not None: # MASK
            w.masked_fill_(padding_mask.all(dim=-1).unsqueeze(1).unsqueeze(2).unsqueeze(3), 0)

        # Softmax(Q*K) * V
        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, query, key, value, padding_mask):
        qkv_same = (query.data_ptr() == key.data_ptr() == value.data_ptr())
        kv_same = (key.data_ptr() == value.data_ptr())

        if qkv_same:
            query, key, value = self.qkv_proj(query).split(self.n_features, dim=-1)
            apply_future_mask = True  # self-attention
        elif kv_same:
            q_w, q_b = self.qkv_proj.weight[:self.n_features, :], self.qkv_proj.bias[:self.n_features]
            query = F.linear(query, q_w, q_b)
            kv_w, kv_b = self.qkv_proj.weight[self.n_features:, :], self.qkv_proj.bias[self.n_features:]
            key, value = F.linear(key, kv_w, kv_b).split(self.n_features, dim=-1)
            apply_future_mask = False
        else:
            assert False

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value, apply_future_mask, padding_mask)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x


class FeedForward(nn.Module):
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def __init__(self, in_features, middle_features, dropout):
        super(FeedForward, self).__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x):
        x = FeedForward.gelu(self.layer_1(x))
        x = self.dropout(x)
        x = self.layer_2(x)

        return x


class SSAttn(nn.Module): # attn from a seq to other seq
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout):
        super(SSAttn, self).__init__()

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout)

        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, p, padding_mask):
        '''x = query or response, p = [p1, padding; p2, padding; ...]'''
        attns = x[0][0][0].data.new(*p.shape).fill_(0)
        for i in range(0, len(p)):
            a = self.attn(p[i], x[0], x[0], x[1])

            full_attn = self.dropout(a)
            xx = self.attn_norm(p[i] + full_attn)

            f = self.ff(xx)
            f = self.dropout(f)
            xx = self.ff_norm(xx + f)

            attns[i] = xx

        return attns


class SeqSelfAttn(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout):
        super(SeqSelfAttn, self).__init__()

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        '''x = query or response, p = [p1, padding; p2, padding; ...]'''
        # word level SA
        # attns = x[0][0][0].data.new(*x.shape).fill_(0)
        a = self.attn(x, x, x, padding_mask)
        full_attn = self.dropout(a)
        x = self.attn_norm(x + full_attn)
        f = self.ff(x)
        f = self.dropout(f)
        x = self.ff_norm(x + f)
        return x

class HierAttn(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout):
        super(HierAttn, self).__init__()

        # self.selfattn = SeqSelfAttn(n_features, n_heads, dropout, attn_dropout, ff_dropout)

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, p, padding_mask):
        '''x = query or response, p = [p1, padding; p2, padding; ...]'''
        # word level SA has been done and fed forward as inputs
        # ax = self.selfattn(x[0], x[1])
        # ap = p[0][0][0].data.new(*p.shape).fill_(0)
        # for i in range(0, len(p)):
        #     a = self.selfattn(p[i], padding_mask[i])
        #     ap[i] = a

        # sentence level (only parse query & target since they have different length)
        # x -> single emb
        mean_x = (x[0].sum(-2)/(~x[1]).sum(-1).unsqueeze(-1)).unsqueeze(-2)

        # TODO: p -> single emb if we want to get a sentence-level attentions
        # mean_ap = ap.sum(-2)/(~padding_mask).sum(-1).unsqueeze(-1)

        # x attn ps
        attns = mean_x[0][0].data.new(p.shape[0], *mean_x.shape).fill_(0)
        for i in range(0, len(p)):
            a = self.attn(mean_x, p[i], p[i], padding_mask[i])

            full_attn = self.dropout(a)
            xx = self.attn_norm(mean_x + full_attn)

            f = self.ff(xx)
            f = self.dropout(f)
            xx = self.ff_norm(xx + f)

            attns[i] = xx

        return attns.squeeze(-2)


class FFAttn(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout):
        super(FFAttn, self).__init__()

        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, x, padding_mask): # x = per
        a = a.permute([1,0,2,3])
        xs = x.permute([1,0,2,3])
        attns = a[0][0][0].data.new(*a.shape).fill_(0)
        
        for i in range(0, len(a)):
            full_attn = self.dropout(a[i])
            x = self.attn_norm(xs[i] + full_attn)

            f = self.ff(x)
            f = self.dropout(f)
            x = self.ff_norm(x + f)

            attns[i] = x

        return attns, padding_mask.permute([1,0,2])


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout, policy):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)
        self.policy = policy

    def forward(self, x, padding_mask, *contexts):
        if self.policy is None or len(contexts)==0: # encode(SA) or original encoder
            '''contexts = [(context1, padding_mask1), ...]'''

            inputs = (x, padding_mask) + contexts

            full_attn = 0
            n_attn = len(inputs) // 2
            for i in range(0, len(inputs), 2):
                c, m = inputs[i], inputs[i+1].bool()
                a = self.attn(x, c, c, m) # first loop: x,x,x i.e. self-attn
                full_attn += (a / n_attn)

            full_attn = self.dropout(full_attn)
            x = self.attn_norm(x + full_attn)

            f = self.ff(x)
            f = self.dropout(f)
            x = self.ff_norm(x + f)

            return (x, padding_mask) + contexts
        elif self.policy == 'add_pad': # separated persona, weighted by weights in context
            full_attn = 0

            # first layer: self-attn
            a = self.attn(x, x, x, padding_mask)
            full_attn += (a / 3)

            # sencond layer: context
            a = self.attn(x, contexts[0], contexts[0], contexts[1].bool())
            full_attn += (a / 3)

            # third layer: separted persona
            for i in range(2, len(contexts)-2, 3):
                p, m, w = contexts[i], contexts[i+1].bool(), contexts[i+2]
                a = self.attn(x, p, p, m)
                # full_attn += (a) / 3
                full_attn += (a*w) / 3 # !!!!!!!ablation
            
            full_attn = self.dropout(full_attn)
            x = self.attn_norm(x + full_attn)

            f = self.ff(x)
            f = self.dropout(f)
            x = self.ff_norm(x + f)

            return (x, padding_mask) + contexts
        elif self.policy == 'hier_sep_attn': # separated persona, weighted by weights in context
            full_attn = 0

            # first layer: self-attn
            a = self.attn(x, x, x, padding_mask)
            full_attn += (a / 3)

            # sencond layer: context
            a = self.attn(x, contexts[0], contexts[0], contexts[1].bool())
            full_attn += (a / 3)

            # third layer: separted persona
            for i in range(2, len(contexts)-2, 3):
                p, m, w = contexts[i], contexts[i+1].bool(), contexts[i+2]
                a = self.attn(x, p, p, m)
                full_attn += (a*w) / 3
            
            full_attn = self.dropout(full_attn)
            x = self.attn_norm(x + full_attn)

            f = self.ff(x)
            f = self.dropout(f)
            x = self.ff_norm(x + f)

            return (x, padding_mask) + contexts


class TransformerModule(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout, 
                 n_segments=None, policy=None):
        super(TransformerModule, self).__init__()

        self.embeddings = nn.Embedding(n_embeddings, embeddings_size, padding_idx=padding_idx)
        self.pos_embeddings = nn.Embedding(n_pos_embeddings + 1, embeddings_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList([TransformerBlock(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout, policy) for _ in range(n_layers)])
        self.n_segments = n_segments        
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, std=0.02)

    def forward(self, x, enc_contexts=[]):
        padding_mask = x.eq(self.embeddings.padding_idx)

        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, self.pos_embeddings.padding_idx)
        
        x = self.embeddings(x) * math.sqrt(self.embeddings.embedding_dim) + self.pos_embeddings(positions)
        x = self.embed_dropout(x)

        enc_contexts = sum(enc_contexts, ())

        if self.n_segments is not None:
            padding_mask = padding_mask.float()  # fucking checkpoint_sequential
            padding_mask.requires_grad_()  # fucking checkpoint_sequential
            out = checkpoint_sequential(self.layers, self.n_segments, x, padding_mask, *enc_contexts)
            x = out[0]
        else:
            for layer in self.layers:
                out = layer(x, padding_mask, *enc_contexts)
                x = out[0]
        
        return x, padding_mask

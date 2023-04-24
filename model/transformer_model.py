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

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_module import TransformerModule, HierAttn, FFAttn, SSAttn


class TransformerModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout,
                 bos_id, eos_id, max_seq_len=256, beam_size=5, sample=False,
                 length_penalty=0.8, annealing_topk=None, annealing=0, 
                 diversity_coef=0, diversity_groups=1, n_segments=None, policy=None):

        super(TransformerModel, self).__init__()

        self.padding_idx = padding_idx
        self.n_embeddings = n_embeddings
        self.n_pos_embeddings = n_pos_embeddings
        self.embeddings_size = embeddings_size

        self.bos_id = bos_id
        self.eos_id = eos_id

        self.max_seq_len = max_seq_len
        self.beam_size = beam_size
        self.sample = sample
        self.length_penalty_coef = length_penalty
        self.annealing = annealing
        self.annealing_topk = annealing_topk
        self.diversity_coef = diversity_coef
        self.diversity_groups = diversity_groups

        if policy == 'add_pad':
            pass
        elif policy == 'hier_sep_attn':
            pass
        else:
            policy = None
        self.policy = policy

        self.transformer_module = TransformerModule(n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                                                    padding_idx, n_heads, dropout, embed_dropout, attn_dropout,
                                                    ff_dropout, n_segments, policy)
        self.pre_softmax = nn.Linear(embeddings_size, n_embeddings, bias=False)
        self.pre_softmax.weight = self.transformer_module.embeddings.weight
        
        self.weight_mlp_1 = nn.Linear(embeddings_size, embeddings_size)
        self.weight_mlp_2 = nn.Linear(embeddings_size, 1)
        self.weight_mlp_3 = nn.Linear(embeddings_size, embeddings_size)
        self.weight_mlp_4 = nn.Linear(embeddings_size, 1)
        nn.init.normal_(self.weight_mlp_1.weight, std=0.02)
        nn.init.normal_(self.weight_mlp_2.weight, std=0.02)
        nn.init.normal_(self.weight_mlp_3.weight, std=0.02)
        nn.init.normal_(self.weight_mlp_4.weight, std=0.02)

        self.pq_sigmoid = nn.Linear(embeddings_size, 1, bias=True)
        self.pq_sigmoid_1 = nn.Linear(embeddings_size*2, embeddings_size, bias=True)
        self.pq_sigmoid_2 = nn.Linear(embeddings_size, 1, bias=True)
        nn.init.xavier_uniform_(self.pq_sigmoid.weight)
        nn.init.xavier_uniform_(self.pq_sigmoid_1.weight)
        nn.init.xavier_uniform_(self.pq_sigmoid_2.weight)

        self.hattn_post = SSAttn(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout)
        self.hattn_pri = SSAttn(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout)

        self.hiattn_post = HierAttn(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout)
        self.hiattn_pri = HierAttn(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout)

        # self.attn_ff = FFAttn(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout)

    def forward(self, x, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        return self.decode(x, enc_contexts)

    def encode(self, x):
        return self.transformer_module(x)

    # 返回中间结果(x,y分别对P的attn)，算KL_D
    def post_attn(self, x, persona_info, m):
        return self.hattn_post(x, persona_info, m)

    def pri_attn(self, x, persona_info, m):
        return self.hattn_pri(x, persona_info, m)

    def post_hier_attn(self, x, persona_info, m):
        return self.hiattn_post(x, persona_info, m)
    
    def pri_hier_attn(self, x, persona_info, m):
        return self.hiattn_pri(x, persona_info, m)
    
    # def attn_to_emb(self, a, p, m):
    #     return self.attn_ff(a, p, m)

    def sentence_attn_to_weight(self, attn):
        w = self.weight_mlp_3(attn)
        weights = self.weight_mlp_4(w).squeeze(-1)
        
        out = torch.sigmoid(weights)
        return out

    def attn_to_weight(self, attn, mask):
        a = attn * (~mask).unsqueeze(-1)
        added = a.sum(-2)
        nums = (~mask).sum(-1).unsqueeze(-1)
        add_attn = added / (nums.float()+1e-5)
        w = self.weight_mlp_1(add_attn)
        weights = self.weight_mlp_2(w).squeeze(-1)
        
        out = torch.sigmoid(weights)
        return out, weights

    def generate(self, enc_x):
        return self.pre_softmax(enc_x)

    def decode(self, x, enc_contexts=[]):
        x, _ = self.transformer_module(x, enc_contexts)
        return self.generate(x)

    def encode_multiple_persona(self, personas, batch_first=False):
        out_dims = personas.permute([1,0,2]).shape
        out_tensor = torch.cuda.FloatTensor(*(out_dims + (self.embeddings_size,))).fill_(0)
        out_mask = torch.cuda.BoolTensor(*out_dims).fill_(True)
        personas = personas.permute([1,0,2]) # batch_size, group_size, sentence_size => group_size, batch_size, sentence_size
        for i in range(0, len(personas)):
            out_tensor[i] = self.transformer_module(personas[i])[0]
            out_mask[i] = self.transformer_module(personas[i])[1]
        if batch_first:
            out_t = out_tensor.permute([1,0,2,3])
            out_m = out_mask.permute([1,0,2])
            return out_t, out_m
        return out_tensor, out_mask

    def weighted_encode_multiple_persona(self, history, persona_info, persona_len):
        enc_his = self.transformer_module(history)
        enc_persona = [self.transformer_module(p) for p in persona_info]
        enc_persona_weighted = []
        out_weights = torch.cuda.FloatTensor()

        for idx in range(len(enc_persona)):
            M_his = torch.mean(enc_his[0][idx], 0)
            M_pers = torch.sum(enc_persona[idx][0],1) / torch.unsqueeze(torch.sum(~enc_persona[idx][1],1),1) # out: num_pers, 768
            C = torch.cat(torch.broadcast_tensors(M_his, M_pers), 1)

            logits_weight = self.pq_sigmoid_2(self.pq_sigmoid_1(C))
            out_weight = torch.sigmoid(logits_weight)

            # enc_persona[idx][0] = torch.mul(enc_persona[idx][0], torch.unsqueeze(out_weight,1))
            enc_persona_weighted.append((torch.mul(enc_persona[idx][0], torch.unsqueeze(out_weight,1)), enc_persona[idx][1]))
            out_weights = torch.cat((out_weights, out_weight))
        

        # persona embedding
        # method 1: concate all persona to create persona
        # method 2: add all persona to create persona
        added_persona = [torch.unsqueeze(torch.sum(per, 0), 0) for (per, pad) in enc_persona_weighted]
        added_persona_pad = [torch.unsqueeze(~(torch.sum(~pad, 0)>0), 0) for (per, pad) in enc_persona_weighted]
        # method 2.5: add all but not weighted
        # added_persona1 = [torch.unsqueeze(torch.sum(per, 0), 0) for (per, pad) in enc_persona]
        # added_persona_pad1 = [torch.unsqueeze(~(torch.sum(~pad, 0)>0), 0) for (per, pad) in enc_persona]
        return (torch.cat(added_persona), torch.cat(added_persona_pad)), torch.squeeze(out_weights)
        # method 3: take all as separated context

    def weighted_encode(self, history, persona_info, persona_len):
        enc_his = self.transformer_module(history)
        enc_persona = self.transformer_module(persona_info)
        batch_weights = []

        # list_persona = torch.split(enc_persona, [i+2 for i in persona_len], dim=0)

        for idx, batch_len in enumerate(persona_len):
            weights = []
            align = 0
            for single_len in batch_len:
                # split persona & calculate weight
                
                # # method 1: concate all
                # C = torch.cat((enc_his[0][idx], enc_persona[0][idx][align:align+single_len+2]),0).to('cuda')
                # weight = torch.mean(torch.squeeze(self.pq_sigmoid_3(self.pq_sigmoid_2(self.pq_sigmoid_1(C))))).to('cuda')
                # weights.append(weight)
                # align += single_len+2

                # method 2: concate last - decrapted because of low-representive features
                # A = torch.add(enc_his[0][idx][-1], enc_persona[0][idx][align+single_len+1]).to('cuda')
                # logits_weight = (self.pq_sigmoid_2(self.pq_sigmoid_1(A))).to('cuda')
                # # logits_weight = self.pq_sigmoid(A).to('cuda')
                # weight_t = torch.sigmoid(logits_weight)
                # weight = torch.squeeze(weight_t).to('cuda')
                # weights.append(weight_t)
                # align += single_len+2
                # # adapt weight to encoded_persona
                # mask_weight = torch.full((single_len+2,), weight.data)
                # masked_weight = torch.ones((persona_info.shape[1]))
                # masked_weight[:single_len+2] = mask_weight
                # enc_persona[0][idx] *= torch.unsqueeze(masked_weight, 1).to('cuda')

                # method 3: mean-pooling & concate
                M_his = torch.mean(enc_his[0][idx], 0)
                M_per = torch.mean(enc_persona[0][idx][align:align+single_len+2], 0)
                A = torch.cat((M_his, M_per)).to('cuda')
                logits_weight = (self.pq_sigmoid_2(self.pq_sigmoid_1(A))).to('cuda')
                # logits_weight = self.pq_sigmoid(A).to('cuda')
                weight_t = torch.sigmoid(logits_weight)
                weight = torch.squeeze(weight_t).to('cuda')
                weights.append(weight_t)
                align += single_len+2
                # adapt weight to encoded_persona
                mask_weight = torch.full((single_len+2,), weight.data)
                masked_weight = torch.ones((persona_info.shape[1]))
                masked_weight[:single_len+2] = mask_weight
                enc_persona[0][idx] *= torch.unsqueeze(masked_weight, 1).to('cuda')
            batch_weights.append(weights)

        # method 3: separated encode personas (since the transformer will leak the msg above)
        
        return enc_persona, batch_weights
        
    def predict(self, contexts=[]):
        self.eval()

        enc_contexts = [self.encode(c) for c in contexts]
        prediction = self.beam_search(enc_contexts)

        return prediction

    def predict_v1(self, persona_info, h, persona_len):
        self.eval()

        context = self.encode(h)
        persona, out_weights = self.weighted_encode(h, persona_info, persona_len)
        enc_contexts = [persona, context]
        prediction = self.beam_search(enc_contexts)

        return prediction, out_weights
    
    def predict_v2(self, persona_info, h, persona_len):
        self.eval()

        context = self.encode(h)
        persona, out_weights = self.weighted_encode_multiple_persona(h, persona_info, persona_len)
        enc_contexts = [persona, context]
        prediction = self.beam_search(enc_contexts)

        return prediction, out_weights

    def predict_v3(self, persona_info, h, attn='pri'): # named add_pad, with postria msg
        self.eval()

        enc_contexts = []

        context = self.encode(h)
        enc_contexts.append(context)

        persona_emb, persona_mask = self.encode_multiple_persona(persona_info)
        if attn == 'pri':
            pri_attn = self.pri_attn(context, persona_emb, persona_mask)
        elif attn == 'post':
            pri_attn = self.post_attn(context, persona_emb, persona_mask)
        flat_bs_size = torch.Size([pri_attn.shape[0]*pri_attn.shape[1]])
        flat_size = flat_bs_size+pri_attn.shape[2:]
        pri_flat = pri_attn.contiguous().view(flat_size)
        sigmoid_weight, weight_out = self.attn_to_weight(pri_flat, persona_mask.view(flat_bs_size+persona_mask.shape[2:]))
        raw_weights = weight_out.view(persona_mask.shape[:2])
        sweis = sigmoid_weight.view(persona_mask.shape[:2])
        
        # FIXME: why softmax after sigmoid could bring on upgrade
        m = torch.nn.Softmax(dim=0)
        # oweis = m(raw_weights)
        oweis = m(sweis)
        weis = oweis.unsqueeze(-1).unsqueeze(-1)
        # weis = raw_weights.unsqueeze(-1).unsqueeze(-1)
        
        # enc_ps = self.attn_to_emb(pri_attn, persona_emb, persona_mask)
        for index, (ori_p, enc_p, mask, attn_weight) in enumerate(zip(persona_info.permute([1,0,2]), persona_emb, persona_mask, weis)):
            enc_contexts.append((enc_p, mask, attn_weight))
        
        prediction = self.beam_search(enc_contexts)

        return prediction, oweis, sweis

    def predict_v4(self, persona_info, h, attn='pri'):
        self.eval()

        enc_contexts = []

        context = self.encode(h)
        enc_contexts.append(context)

        persona_emb, persona_mask = self.encode_multiple_persona(persona_info)
        if attn == 'pri':
            pri_attn = self.pri_hier_attn(context, persona_emb, persona_mask)
        elif attn == 'post':
            pri_attn = self.post_hier_attn(context, persona_emb, persona_mask)
        flat_bs_size = torch.Size([pri_attn.shape[0]*pri_attn.shape[1]])
        flat_size = flat_bs_size+pri_attn.shape[2:]
        pri_flat = pri_attn.contiguous().view(flat_size)
        weight_out = self.sentence_attn_to_weight(pri_flat)
        raw_weights = weight_out.view(persona_mask.shape[:2])
        # m = torch.nn.Softmax(dim=0)
        # weis = m(raw_weights)
        # weis = weis.unsqueeze(-1).unsqueeze(-1)
        weis = raw_weights.unsqueeze(-1).unsqueeze(-1)
        
        # enc_ps = self.attn_to_emb(pri_attn, persona_emb, persona_mask)
        for index, (ori_p, enc_p, mask, attn_weight) in enumerate(zip(persona_info.permute([1,0,2]), persona_emb, persona_mask, weis)):
            enc_contexts.append((enc_p, mask, attn_weight))
        
        prediction = self.beam_search(enc_contexts)

        return prediction, weight_out

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

    def beam_search(self, enc_contexts=[], return_beams=False):
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long, device=device)
            
            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.bool, device=device)

            beam_enc_contexts = []
            if self.policy == 'add_pad' \
                or self.policy == 'hier_sep_attn':
                for c, p in enc_contexts:
                    c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                    c = c.view(-1, c.shape[2], c.shape[3])
                    p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
                    p = p.view(-1, p.shape[2])
                    beam_enc_contexts.append((c, p))
                    break
                for i in range(1, len(enc_contexts)):
                    c, p, w = enc_contexts[i]
                    c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                    c = c.view(-1, c.shape[2], c.shape[3])
                    p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
                    p = p.view(-1, p.shape[2])
                    w = w.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                    w = w.view(-1, w.shape[2], w.shape[3])
                    beam_enc_contexts.append((c, p, w))
            else:
                for c, p in enc_contexts:
                    c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                    c = c.view(-1, c.shape[2], c.shape[3])
                    p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
                    p = p.view(-1, p.shape[2])
                    beam_enc_contexts.append((c, p))
            
            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)

            for i in range(self.max_seq_len):
                outputs, _ = self.transformer_module(prevs, beam_enc_contexts)

                logits = self.generate(outputs[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.view(batch_size, self.beam_size, -1)

                beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
                penalty = penalty.unsqueeze(-1).repeat(1, 1, self.n_embeddings)
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:
                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        if random.random() < current_sample_prob:
                            beam_probas = F.softmax(g_beam_scores, dim=-1)
                            if self.annealing_topk is not None:
                                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                                g_idxs = torch.multinomial(beam_probas, group_size)
                                g_idxs = torch.gather(sample_idxs, 1, g_idxs)
                            else:
                                g_idxs = torch.multinomial(beam_probas, group_size)
                        else:
                            _, g_idxs = g_beam_scores.topk(group_size, dim=-1)
                        
                        g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
                        g_idxs += g * group_size * self.n_embeddings

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, self.n_embeddings), torch.ones((batch_size, group_size), device=device))
                     
                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1) 

                    beam_idxs = (idxs.float() / self.n_embeddings).long()

                penalty = torch.gather(penalty, 1, idxs)
                sym_idxs = torch.fmod(idxs, log_probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                sym_idxs[is_end] = self.padding_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                if all(is_end.view(-1)):
                    break
                
                beam_scores *= penalty
                current_sample_prob *= self.annealing

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)

            if return_beams:
                 return result, beam_lens

            if self.sample:
                probs = F.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)
            
            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len-1]
                predicts.append(best_seq.tolist())
                
        return predicts

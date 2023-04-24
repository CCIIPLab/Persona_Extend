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

from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import pad_sequence, pad_sequence_of_sequence
from .optim import Adam, NoamOpt
from .loss import LabelSmoothingLoss

from tensorboardX import SummaryWriter

# from prefetch_generator import BackgroundGenerator
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

class Trainer:
    def __init__(self, model, train_dataset, test_dataset=None, batch_size=8,
                 batch_split=1, lm_weight=0.5, risk_weight=0, lr=6.25e-5, lr_warmup=2000, 
                 n_jobs=0, clip_grad=None, label_smoothing=0, device=torch.device('cuda'),
                 ignore_idxs=[], int_label=False, persona_enc_policy='concate', log_name='',
                 freeze_posteria_attn=False):
        self.writer = SummaryWriter('log/'+persona_enc_policy+'/'+log_name)
        self.global_train_step = 0
        self.global_test_step = 0
        
        self.model = model.to(device)
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx).to(device)
        self.mse_criterion = nn.MSELoss().to(device)
        self.cos_criterion = nn.CosineEmbeddingLoss().to(device)
        # self.binary_criterion = nn.CrossEntropyLoss().to(device)
        self.binary_criterion = nn.BCELoss().to(device) # reduce=False
        # self.binary_criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.FloatTensor([1., 42.])).to(device)
        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing, ignore_index=self.model.padding_idx).to(device)
        
        if freeze_posteria_attn:
            for p in self.model.hattn_post.parameters():
                p.requires_grad=False
            for p in self.model.hattn_pri.parameters():
                p.requires_grad=False
            for p in self.model.weight_mlp_1.parameters():
                p.requires_grad=False
            for p in self.model.weight_mlp_2.parameters():
                p.requires_grad=False

        base_optimizer = Adam(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.embeddings_size, lr, lr_warmup, base_optimizer)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size//batch_split, shuffle=True, 
                                           num_workers=8, collate_fn=self.collate_func,)
        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size//batch_split, shuffle=False, 
                                            num_workers=2, collate_fn=self.collate_func)

        self.batch_split = batch_split
        self.lm_weight = lm_weight
        self.risk_weight = risk_weight
        self.clip_grad = clip_grad
        self.device = device
        self.ignore_idxs = ignore_idxs
        self.persona_enc_policy = persona_enc_policy
        self.int_label = int_label
        
        self.cons_1 = torch.cuda.FloatTensor([1])
        self.padding_sentence = train_dataset.vocab.info_eos_id

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)
        # for var_name in self.optimizer.state_dict():
        #     print(var_name, "\t", self.optimizer.state_dict()[var_name])
        # for var_name in state_dict['optimizer']:
        #     print(var_name, "\t", state_dict['optimizer'][var_name])
        # self.optimizer.load_state_dict(state_dict['optimizer'])
        # FIXME: now disable the reload of optimizer

    def collate_func(self, data):
        persona_info, h, y, w, persona_len = zip(*data)

        contexts = []

        if max(map(len, persona_info)) > 0:
            if self.persona_enc_policy == 'concate' or \
               self.persona_enc_policy == 'link' or \
               self.persona_enc_policy == 'sep':
                # TODO: filter all-0
                persona_info = [torch.tensor(d, dtype=torch.long) for d in persona_info]
                persona_info = pad_sequence(persona_info, batch_first=True, padding_value=self.model.padding_idx)
                contexts.append(persona_info)

            elif self.persona_enc_policy == 'add':
                # TODO: set a frozen size of persona_info
                max_len = 0
                persona_info_sep = []
                for len_list, aggr_persona in zip(persona_len, persona_info):
                    start_index = 0
                    aggr_persona_2sep = []
                    for pl in len_list:
                        aggr_persona_2sep.append(aggr_persona[start_index:start_index+pl+2])
                        start_index += (pl+2)
                        max_len = max(max_len, pl+2)
                    persona_info_sep.append(aggr_persona_2sep)

                persona_info = [[torch.tensor(d, dtype=torch.long) for d in batch_weights] for batch_weights in persona_info_sep]
                persona_info = [pad_sequence(y, batch_first=True, padding_value=self.model.padding_idx, max_len=max_len) for y in persona_info]
                # parsed_persona_info = []
                # for info in persona_info:
                #     info = [torch.tensor(d, dtype=torch.long) for d in info]
                #     info = pad_sequence(info, batch_first=True, padding_value=self.model.padding_idx)
                #     parsed_persona_info.append(info)
                # persona_info = parsed_persona_info
                
                # contexts.append(persona_info)
            elif self.persona_enc_policy == 'add_pad' or self.persona_enc_policy == 'hier_sep_attn':
                p = [[torch.tensor(p, dtype=torch.long) for p in per] for per in persona_info]
                persona_info = pad_sequence_of_sequence(p, batch_first=True, padding_value=self.model.padding_idx, padding_sentence_value=self.padding_sentence)

        if max(map(len, h)) > 0:
            h = [torch.tensor(d, dtype=torch.long) for d in h]
            h = pad_sequence(h, batch_first=True, padding_value=self.model.padding_idx)
            # contexts.append(h)
    
        y = [torch.tensor(d, dtype=torch.long) for d in y]
        y = pad_sequence(y, batch_first=True, padding_value=self.model.padding_idx)

        if self.int_label is True:
            if self.persona_enc_policy == 'add_pad' or self.persona_enc_policy == 'hier_sep_attn':
                w = [torch.tensor(d, dtype=torch.int) for d in w]
                w = pad_sequence(w, batch_first=True, padding_value=self.model.padding_idx).type(torch.FloatTensor)
            else:
                w = [[torch.tensor(d, dtype=torch.int) for d in batch_weights] for batch_weights in w]
        else:
            w = [[torch.tensor(d, dtype=torch.float) for d in batch_weights] for batch_weights in w]

        # return contexts, y, w
        return persona_info, h, y, w, persona_len

    def to_cuda(self, persona_info, history, targets, weights, persona_len):
        # To CUDA
        history = history.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        # weight = weight.to(self.device)
        if self.persona_enc_policy == 'add_pad' or \
            self.persona_enc_policy == 'hier_sep_attn':
            weight = weights.to(self.device, non_blocking=True)
        else:
            weight = [[tensor.to(self.device) for tensor in batch_weights] for batch_weights in weights]
            
        if self.persona_enc_policy == 'concate' or \
            self.persona_enc_policy == 'link' or \
            self.persona_enc_policy == 'sep' or \
            self.persona_enc_policy == 'hier_sep_attn' or \
            self.persona_enc_policy == 'add_pad':
            persona_info = persona_info.to(self.device)
        elif self.persona_enc_policy == 'add':
            persona_info = [p.to(self.device, non_blocking=True) for p in persona_info]
            # persona_output = []
            # for persona in persona_info:
            #     persona = [p.to(self.device) for p in persona]
            #     persona_output.append(persona)
            # persona_info = persona_output

        return persona_info, history, targets, weight, persona_len


    def _eval_train(self, epoch, risk_func=None):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        lm_loss = 0
        weight_loss = 0
        risk_loss = 0
        cos_loss = 0

        for i, (persona_info, history, targets, weights, persona_len) in enumerate(tqdm_data): # use collate_func above, pad persona & history, then dump to context
        # for  (persona_info, history, targets, weights, persona_len) in self.train_dataloader: # use collate_func above, pad persona & history, then dump to context
            # contexts = [persona_info, history]
            # contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)
            
            persona_info, history, targets, weight, persona_len = self.to_cuda(persona_info, history, targets, weights, persona_len)

            enc_contexts = []

            # initialize lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)

            
            # for context in contexts:# [persona, history]
            #     # TODO: separate persona & history
            #     # TODO: persona embedding X contexts -> weights, weights -> (1) new loss, (2) update enc_contexts
            #     enc_context = self.model.encode(context.clone())
            #     enc_contexts.append(enc_context)
                
            #     if self.lm_weight > 0:
            #         context_outputs = self.model.generate(enc_context[0])
            #         ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
            #         context.masked_fill_(ignore_mask, self.model.padding_idx)
            #         prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
            #         batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / len(contexts))

            
            # persona-query weighted loss (mse or cross)
            batch_weight_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            
            # history
            # FIXME: change enc_context to enc_history, etc
            enc_context = self.model.encode(history.clone())
            enc_contexts.append(enc_context)

            # Part of LM loss
            if self.lm_weight > 0:
                context_outputs = self.model.generate(enc_context[0])
                ignore_mask = torch.stack([history == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                history.masked_fill_(ignore_mask, self.model.padding_idx)
                prevs, nexts = context_outputs[:, :-1, :].contiguous(), history[:, 1:].contiguous()
                batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / 2)

            # target
            enc_target = self.model.encode(targets.clone())

            # persona
            if self.persona_enc_policy == 'concate':
                enc_context = self.model.encode(persona_info.clone())

                enc_contexts.append(enc_context)
            
                if self.lm_weight > 0:
                    context_outputs = self.model.generate(enc_context[0])
                    ignore_mask = torch.stack([persona_info == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                    persona_info.masked_fill_(ignore_mask, self.model.padding_idx)
                    prevs, nexts = context_outputs[:, :-1, :].contiguous(), persona_info[:, 1:].contiguous()
                    batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / 2)


            elif self.persona_enc_policy == 'link':
                enc_context, weight_out = self.model.weighted_encode(history.clone(), persona_info.clone(), persona_len) # FIXME: check weighted_encode

                count = 0
                if self.int_label is False:
                    weight_criterion = self.mse_criterion
                    for out, label in zip(weight_out, weight):
                        for exact_out, exact_label in zip(out,label):
                            count += 1
                            batch_weight_loss += weight_criterion(exact_out, exact_label)
                    # batch_weight_loss *= 100 # FIXME: will it work to focus weight_loss?
                    batch_weight_loss /= count
                else:
                    weight_criterion = self.binary_criterion

                    tmp_len = 0
                    for tmp_item in weight_out:
                        tmp_len += len(tmp_item)
                    a = torch.zeros(tmp_len).to(self.device)

                    add_index = 0
                    for tmp_item in weight_out:
                        for t in tmp_item:
                            l = torch.zeros(add_index).to(self.device)
                            r = torch.zeros(tmp_len - add_index - 1).to(self.device)
                            a += torch.cat((l, t, r))
                            add_index += 1

                    # a = torch.cuda.FloatTensor(sum(weight_out, []))
                    # a.requires_grad_()
                    b = torch.FloatTensor(sum(weight, [])).to(self.device)
                    # 
                    batch_weight_loss = weight_criterion(a, b)
                    # Weight
                    # pos_weight = torch.tensor([1., 10.]) # [1, 5] fits ground_truth
                    # weight_ = pos_weight[b.data.view(-1).long()].view_as(b).to(self.device)
                    # batch_weight_loss = weight_criterion(a, b) * weight_
                    # batch_weight_loss = batch_weight_loss.mean()     

                enc_contexts.append(enc_context)
            
                if self.lm_weight > 0:
                    context_outputs = self.model.generate(enc_context[0])
                    ignore_mask = torch.stack([persona_info == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                    persona_info.masked_fill_(ignore_mask, self.model.padding_idx)
                    prevs, nexts = context_outputs[:, :-1, :].contiguous(), persona_info[:, 1:].contiguous()
                    batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / 2)
           

            elif self.persona_enc_policy == 'add':
                # ---------------------------------------
                persona_info = [p.clone() for p in persona_info]
                enc_context, weight_out = self.model.weighted_encode_multiple_persona(history.clone(), persona_info, persona_len)
                labels = torch.FloatTensor(sum(weight, [])).to(self.device)
                batch_weight_loss = self.binary_criterion(weight_out, labels)
                
                enc_contexts.append(enc_context)

                # for index in range(len(persona_info)):
                #     if self.lm_weight > 0:
                #         context_outputs = self.model.generate(enc_context[0])
                #         ignore_mask = torch.stack([persona_info[index] == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                #         persona_info[index].masked_fill_(ignore_mask, self.model.padding_idx)
                #         prevs, nexts = context_outputs[:, :-1, :].contiguous(), persona_info[index][:, 1:].contiguous()
                #         batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / 2)

            elif self.persona_enc_policy == 'add_pad':
                persona_info = persona_info.clone()
                persona_emb, persona_mask = self.model.encode_multiple_persona(persona_info)
                pri_attn = self.model.pri_attn(enc_context, persona_emb, persona_mask)
                post_attn = self.model.post_attn(enc_target, persona_emb, persona_mask)
                
                # KL-DISTANCE LOSS
                pass

                # COS LOSS
                flat_bs_size = torch.Size([pri_attn.shape[0]*pri_attn.shape[1]])
                flat_size = flat_bs_size+pri_attn.shape[2:]
                pri_flat = pri_attn.contiguous().view(flat_size)
                post_flat = post_attn.contiguous().view(flat_size)
                batch_cos_loss = self.cos_criterion(pri_flat, post_flat, self.cons_1) # pos_l = torch.ones(flat_bs_size).to(self.device)

                # Attn to weight
                flat_weight = weight.permute([1,0]).contiguous().view([-1])
                weight_out_sigmoid, weight_out = self.model.attn_to_weight(post_flat, persona_mask.view(flat_bs_size+persona_mask.shape[2:]))
                
                batch_weight_loss = self.binary_criterion(weight_out_sigmoid, flat_weight)

                # append attns
                # enc_ps = self.model.attn_to_emb(post_attn, persona_emb, persona_mask)
                raw_weights = weight_out.view(persona_mask.shape[:2])
                m = torch.nn.Softmax(dim=0)
                weis = m(raw_weights)
                weis = weis.unsqueeze(-1).unsqueeze(-1)
                # weis = raw_weights.unsqueeze(-1).unsqueeze(-1)
                for index, (ori_p, enc_p, mask, attn_weight) in enumerate(zip(persona_info.permute([1,0,2]), persona_emb, persona_mask, weis)):
                    enc_contexts.append((enc_p, mask, attn_weight))
                    
                    if self.lm_weight > 0:
                        context_outputs = self.model.generate(enc_p)
                        ignore_mask = torch.stack([enc_p == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                        enc_p = enc_p.masked_fill(ignore_mask, self.model.padding_idx)
                        prevs, nexts = context_outputs[:, :-1, :].contiguous(), ori_p[:, 1:].contiguous()
                        batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / (2 * len(persona_emb)))
                
            elif self.persona_enc_policy == 'sep':
                pass

            elif self.persona_enc_policy == 'hier_sep_attn':
                persona_info = persona_info.clone()
                persona_emb, persona_mask = self.model.encode_multiple_persona(persona_info)
                pri_attn = self.model.pri_hier_attn(enc_context, persona_emb, persona_mask)
                post_attn = self.model.post_hier_attn(enc_target, persona_emb, persona_mask)
                
                # KL-DISTANCE LOSS
                pass

                # COS LOSS
                flat_bs_size = torch.Size([pri_attn.shape[0]*pri_attn.shape[1]])
                flat_size = flat_bs_size+pri_attn.shape[2:]
                pri_flat = pri_attn.contiguous().view(flat_size)
                post_flat = post_attn.contiguous().view(flat_size)
                batch_cos_loss = self.cos_criterion(pri_flat, post_flat, self.cons_1) # pos_l = torch.ones(flat_bs_size).to(self.device)

                # Attn to weight
                flat_weight = weight.permute([1,0]).contiguous().view([-1])
                weight_out = self.model.sentence_attn_to_weight(post_flat)
                
                batch_weight_loss = self.binary_criterion(weight_out, flat_weight)

                # append attns
                # enc_ps = self.model.attn_to_emb(post_attn, persona_emb, persona_mask)
                raw_weights = weight_out.view(persona_mask.shape[:2])
                # m = torch.nn.Softmax(dim=0)
                # weis = m(raw_weights)
                # weis = weis.unsqueeze(-1).unsqueeze(-1)
                weis = raw_weights.unsqueeze(-1).unsqueeze(-1)
                for index, (ori_p, enc_p, mask, attn_weight) in enumerate(zip(persona_info.permute([1,0,2]), persona_emb, persona_mask, weis)):
                    enc_contexts.append((enc_p, mask, attn_weight))
                    
                    if self.lm_weight > 0:
                        context_outputs = self.model.generate(enc_p)
                        ignore_mask = torch.stack([enc_p == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                        enc_p = enc_p.masked_fill(ignore_mask, self.model.padding_idx)
                        prevs, nexts = context_outputs[:, :-1, :].contiguous(), ori_p[:, 1:].contiguous()
                        batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / (2 * len(persona_emb)))
                

            
            # assembled (persona, context) need NOT ordered
            

            # s2s loss
            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
            
            # risk loss
            batch_risk_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            if risk_func is not None and self.risk_weight > 0:

                beams, beam_lens = self.model.beam_search(enc_contexts, return_beams=True)

                target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
                targets = [target[1:length-1].tolist() for target, length in zip(targets, target_lens)]
                batch_risks = []
                for b in range(beams.shape[1]):
                    predictions = [b[1:l-1].tolist() for b, l in zip(beams[:, b, :], beam_lens[:, b])]
                    risks = torch.tensor(risk_func(predictions, targets), dtype=torch.float, device=self.device)
                    batch_risks.append(risks)
                batch_risks = torch.stack(batch_risks, dim=-1)

                batch_probas = []
                for b in range(beams.shape[1]):
                    logits = self.model.decode(beams[:, b, :-1], enc_contexts)
                    probas = F.log_softmax(logits, dim=-1)
                    probas = torch.gather(probas, -1, beams[:, b, 1:].unsqueeze(-1)).squeeze(-1)
                    probas = probas.sum(dim=-1) / beam_lens[:, b].float()
                    batch_probas.append(probas)
                batch_probas = torch.stack(batch_probas, dim=-1)
                batch_probas = F.softmax(batch_probas, dim=-1)
                
                batch_risk_loss = torch.mean((batch_risks * batch_probas).sum(dim=-1))
            
            # DEBUGING
            # outputs.register_hook(lambda grad: print('outputs', grad))
            # batch_loss.register_hook(lambda grad: print('batch_loss', grad))
            
            # batch_lm_loss.register_hook(lambda grad: print('batch_lm_loss', grad))

            # batch_weight_loss.register_hook(lambda grad: print('batch_weight_loss', grad))
            # a.register_hook(lambda grad: print('weight_out_a', grad))
            # weight_out[0][0].register_hook(lambda grad: print('real_weight_00', grad))



            # optimization
            # (1000.0 * batch_weight_loss).backward()
            # (0.001* batch_weight_loss).backward()
            # batch_cos_loss.backward()
            # batch_weight_loss.backward()
            # !!!!!!!ablation: full_loss = (0.3*batch_cos_loss + 0.3*batch_weight_loss + batch_lm_loss * self.lm_weight + self.risk_weight * batch_risk_loss + batch_loss) / self.batch_split
            full_loss = (0.3*batch_cos_loss + batch_lm_loss * self.lm_weight + self.risk_weight * batch_risk_loss + batch_loss) / self.batch_split
            # full_loss = ( batch_weight_loss + batch_lm_loss * self.lm_weight + self.risk_weight * batch_risk_loss + batch_loss) / self.batch_split
            full_loss.backward()

            # TODO: use hyper para to cutdown cos_loss.back (decra too soon by default)

            # gradient accumulation
            if (i + 1) % self.batch_split == 0:
                if self.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

                # with open('./tmp_softmax_tensor', 'w')as f:
                #     tensor2txt = self.model.pq_sigmoid.weight.tolist()
                #     for dim in tensor2txt:
                #         for number in dim:
                #             f.write(str(number)+'\n')

                cur_lr = self.optimizer.step()
                self.optimizer.zero_grad()
                self.writer.add_scalar('Train/learning_rate', cur_lr, i+self.global_train_step)

                # with open('./tmp_softmax_tensor_2', 'w')as f:
                #     tensor2txt = self.model.pq_sigmoid.weight.tolist()
                #     for dim in tensor2txt:
                #         for number in dim:
                #             f.write(str(number)+'\n')

            self.writer.add_scalar('Train/loss/cos', batch_cos_loss.item(), i+self.global_train_step)
            self.writer.add_scalar('Train/loss/weight', batch_weight_loss.item(), i+self.global_train_step)
            self.writer.add_scalar('Train/loss/language_model', batch_lm_loss.item(), i+self.global_train_step)
            self.writer.add_scalar('Train/loss/decoder', batch_loss.item(), i+self.global_train_step)
            self.writer.add_scalar('Train/loss/risk', batch_risk_loss.item(), i+self.global_train_step)
            self.writer.add_scalar('Train/loss/full', full_loss.item(), i+self.global_train_step)

            cos_loss = (i * cos_loss + batch_cos_loss.item()) / (i + 1)
            weight_loss = (i * weight_loss + batch_weight_loss.item()) / (i + 1)
            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            risk_loss = (i * risk_loss + batch_risk_loss.item()) / (i + 1)

            tqdm_data.set_postfix({'lm': lm_loss, 'it_cos': batch_cos_loss.item(), 'cos': cos_loss, 'it_wei': batch_weight_loss.item(), 'wei': weight_loss, 'loss': loss,'risk_loss': risk_loss})
        
        self.global_train_step += i

    def _eval_test(self, metric_funcs={}):
        self.model.eval()

        tqdm_data = tqdm(self.test_dataloader, desc='Test')
        loss = 0
        lm_loss = 0
        weight_loss = 0
        distance_loss = 0
        metrics = {name: 0 for name in metric_funcs.keys()}
        # for i, (contexts, targets) in enumerate(tqdm_data):
            # contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)
        for i, (persona_info, history, targets, weights, persona_len) in enumerate(tqdm_data): # use collate_func above, pad persona & history, then dump to context
            
            persona_info, history, targets, weight, persona_len = self.to_cuda(persona_info, history, targets, weights, persona_len)
            
            enc_contexts = []

            # lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            
            # persona-query weighted loss (mse or cross)
            batch_weight_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            
            # history
            enc_context = self.model.encode(history.clone())
            enc_contexts.append(enc_context)

            if self.lm_weight > 0:
                context_outputs = self.model.generate(enc_context[0])
                ignore_mask = torch.stack([history == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                history.masked_fill_(ignore_mask, self.model.padding_idx)
                prevs, nexts = context_outputs[:, :-1, :].contiguous(), history[:, 1:].contiguous()
                batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / 2)
            
            enc_target = self.model.encode(targets.clone())

            # persona
            if self.persona_enc_policy == 'concate':
                enc_context = self.model.encode(persona_info.clone())

            elif self.persona_enc_policy == 'link':
                enc_context, weight_out = self.model.weighted_encode(history.clone(), persona_info.clone(), persona_len) # FIXME: check weighted_encode

                count = 0
                if self.int_label is False:
                    weight_criterion = self.mse_criterion
                    for out, label in zip(weight_out, weight):
                        for exact_out, exact_label in zip(out,label):
                            count += 1
                            batch_weight_loss += weight_criterion(exact_out, exact_label)
                    # batch_weight_loss *= 100 # FIXME: will it work to focus weight_loss?
                    batch_weight_loss /= count
                else:
                    weight_criterion = self.binary_criterion
                    a = torch.FloatTensor(sum(weight_out, [])).to(self.device)#output
                    b = torch.FloatTensor(sum(weight, [])).to(self.device)#label
                    # batch_weight_loss = weight_criterion(a, b)
                    pos_weight = torch.tensor([1., 10.])
                    weight_ = pos_weight[b.data.view(-1).long()].view_as(b).to(self.device)
                    batch_weight_loss = weight_criterion(a, b) * weight_
                    batch_weight_loss = batch_weight_loss.mean()                

            elif self.persona_enc_policy == 'add':
                # ---------------------------------------
                persona_info = [p.clone() for p in persona_info]
                enc_context, weight_out = self.model.weighted_encode_multiple_persona(history.clone(), persona_info, persona_len)
                labels = torch.FloatTensor(sum(weight, [])).to(self.device)
                batch_weight_loss = self.binary_criterion(weight_out, labels)
                
                enc_contexts.append(enc_context)

                # for index in range(len(persona_info)):
                #     if self.lm_weight > 0:
                #         context_outputs = self.model.generate(enc_context[0])
                #         ignore_mask = torch.stack([persona_info[index] == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                #         persona_info[index].masked_fill_(ignore_mask, self.model.padding_idx)
                #         prevs, nexts = context_outputs[:, :-1, :].contiguous(), persona_info[index][:, 1:].contiguous()
                #         batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / 2)

            elif self.persona_enc_policy == 'add_pad':
                persona_info = persona_info.clone()
                persona_emb, persona_mask = self.model.encode_multiple_persona(persona_info)
                pri_attn = self.model.pri_attn(enc_context, persona_emb, persona_mask)
                post_attn = self.model.post_attn(enc_target, persona_emb, persona_mask)
                
                # KL-DISTANCE LOSS
                pass

                # COS LOSS
                flat_bs_size = torch.Size([pri_attn.shape[0]*pri_attn.shape[1]])
                flat_size = flat_bs_size+pri_attn.shape[2:]
                pri_flat = pri_attn.contiguous().view(flat_size)
                post_flat = post_attn.contiguous().view(flat_size)
                batch_cos_loss = self.cos_criterion(pri_flat, post_flat, self.cons_1) # pos_l = torch.ones(flat_bs_size).to(self.device)

                # Attn to weight
                flat_weight = weight.permute([1,0]).contiguous().view([-1])
                weight_out_sigmoid, weight_out = self.model.attn_to_weight(post_flat, persona_mask.view(flat_bs_size+persona_mask.shape[2:]))
                
                batch_weight_loss = self.binary_criterion(weight_out_sigmoid, flat_weight)

                # append attns
                # enc_ps = self.model.attn_to_emb(post_attn, persona_emb, persona_mask)
                raw_weights = weight_out.view(persona_mask.shape[:2])
                m = torch.nn.Softmax(dim=0)
                weis = m(raw_weights)
                weis = weis.unsqueeze(-1).unsqueeze(-1)
                # weis = raw_weights.unsqueeze(-1).unsqueeze(-1)
                for index, (ori_p, enc_p, mask, attn_weight) in enumerate(zip(persona_info.permute([1,0,2]), persona_emb, persona_mask, weis)):
                    enc_contexts.append((enc_p, mask, attn_weight))
                    
                    if self.lm_weight > 0:
                        context_outputs = self.model.generate(enc_p)
                        ignore_mask = torch.stack([enc_p == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                        enc_p = enc_p.masked_fill(ignore_mask, self.model.padding_idx)
                        prevs, nexts = context_outputs[:, :-1, :].contiguous(), ori_p[:, 1:].contiguous()
                        batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / (2 * len(persona_emb)))

            elif self.persona_enc_policy == 'sep':
                pass

            elif self.persona_enc_policy == 'hier_sep_attn':
                persona_info = persona_info.clone()
                persona_emb, persona_mask = self.model.encode_multiple_persona(persona_info)
                pri_attn = self.model.pri_hier_attn(enc_context, persona_emb, persona_mask)
                post_attn = self.model.post_hier_attn(enc_target, persona_emb, persona_mask)
                
                # KL-DISTANCE LOSS
                pass

                # COS LOSS
                flat_bs_size = torch.Size([pri_attn.shape[0]*pri_attn.shape[1]])
                flat_size = flat_bs_size+pri_attn.shape[2:]
                pri_flat = pri_attn.contiguous().view(flat_size)
                post_flat = post_attn.contiguous().view(flat_size)
                batch_cos_loss = self.cos_criterion(pri_flat, post_flat, self.cons_1) # pos_l = torch.ones(flat_bs_size).to(self.device)

                # Attn to weight
                flat_weight = weight.permute([1,0]).contiguous().view([-1])
                weight_out = self.model.sentence_attn_to_weight(post_flat)
                
                batch_weight_loss = self.binary_criterion(weight_out, flat_weight)

                # append attns
                # enc_ps = self.model.attn_to_emb(post_attn, persona_emb, persona_mask)
                raw_weights = weight_out.view(persona_mask.shape[:2])
                # m = torch.nn.Softmax(dim=0)
                # weis = m(raw_weights)
                # weis = weis.unsqueeze(-1).unsqueeze(-1)
                weis = raw_weights.unsqueeze(-1).unsqueeze(-1)
                for index, (ori_p, enc_p, mask, attn_weight) in enumerate(zip(persona_info.permute([1,0,2]), persona_emb, persona_mask, weis)):
                    enc_contexts.append((enc_p, mask, attn_weight))
                    
                    if self.lm_weight > 0:
                        context_outputs = self.model.generate(enc_p)
                        ignore_mask = torch.stack([enc_p == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                        enc_p = enc_p.masked_fill(ignore_mask, self.model.padding_idx)
                        prevs, nexts = context_outputs[:, :-1, :].contiguous(), ori_p[:, 1:].contiguous()
                        batch_lm_loss += (self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / (2 * len(persona_emb)))
             

            # s2s loss
            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
            
            predictions = self.model.beam_search(enc_contexts)
            target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
            targets = [t[1:l-1].tolist() for t, l in zip(targets, target_lens)]
            
            full_loss = batch_cos_loss + batch_weight_loss + batch_lm_loss + batch_loss
            self.writer.add_scalar('Test/loss/cos', batch_cos_loss.item(), i+self.global_test_step)
            self.writer.add_scalar('Test/loss/weight', batch_weight_loss.item(), i+self.global_test_step)
            self.writer.add_scalar('Test/loss/language_model', batch_lm_loss.item(), i+self.global_test_step)
            self.writer.add_scalar('Test/loss/decoder', batch_loss.item(), i+self.global_test_step)
            self.writer.add_scalar('Test/loss/full', full_loss.item(), i+self.global_test_step)

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            weight_loss = (i * weight_loss + batch_weight_loss.item()) / (i + 1)
            distance_loss = (i * distance_loss + batch_cos_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            for name, func in metric_funcs.items():
                score = func(predictions, targets)
                metrics[name] = (metrics[name] * i + score) / (i + 1)

            tqdm_data.set_postfix(dict({'lm_loss': lm_loss, 'cos_loss': distance_loss, 'weight_loss': weight_loss, 'loss': loss}, **metrics))

        self.global_test_step += i
    
    def test(self, metric_funcs={}):
        if hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs)
    
    def call_funcs_bypass(self, funcs=[]):
        for func in funcs:
            func(0)

    def train(self, epochs, after_epoch_funcs=[], risk_func=None):
        for epoch in range(epochs):
            self._eval_train(epoch, risk_func)

            for func in after_epoch_funcs:
                func(epoch)

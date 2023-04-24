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
from torch.utils.data import Dataset
from .text import BPEVocab


class FacebookDataset(Dataset):
    @staticmethod
    def parse_data(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = []
            for line in file.readlines():
                line = line.strip()

                if len(line) == 0:
                    continue

                if line[0] == '0':
                    data.append({'persona_info': [], 'dialog': [], 'weight': []})
                    continue

                dialog_line = line.split('\t')[1:]
                dialog_line = [l.strip() for l in dialog_line]

                if dialog_line[1].startswith('your persona:'):
                    persona_info = dialog_line[1].replace('your persona: ', '')
                    data[-1]['persona_info'].append(persona_info)

                elif len(dialog_line) > 1:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])
                    data[-1]['weight'].append(None)
                    data[-1]['weight'].append([float(item) for item in dialog_line[2].strip('[]').split(', ')])

            return data

            # data = []
            # for line in file.readlines():
            #     line = line.strip()

            #     if len(line) == 0:
            #         continue

            #     space_idx = line.find(' ')
            #     if space_idx == -1:
            #         dialog_idx = int(line)
            #     else:
            #         dialog_idx = int(line[:space_idx])

            #     if int(dialog_idx) == 1:
            #         data.append({'persona_info': [], 'dialog': []})

            #     dialog_line = line[space_idx + 1:].split('\t')
            #     dialog_line = [l.strip() for l in dialog_line]

            #     if dialog_line[0].startswith('your persona:'):
            #         persona_info = dialog_line[0].replace('your persona: ', '')
            #         data[-1]['persona_info'].append(persona_info)

            #     elif len(dialog_line) > 1:
            #         data[-1]['dialog'].append(dialog_line[0])
            #         data[-1]['dialog'].append(dialog_line[1])

            # return data

    @staticmethod
    def parse_float_weight(data):
        for chat in data:
            for index, weight in enumerate(chat['weight']):
                if weight is not None:
                    chat['weight'][index] = [float(i) for i in chat['weight'][index]]
        return data
    
    @staticmethod
    def parse_int_weight(data):
        for chat in data:
            for index, weight in enumerate(chat['weight']):
                if weight is not None:
                    chat['weight'][index] = [int(i) for i in chat['weight'][index]]
        return data

    @staticmethod
    def make_dataset(data, vocab, max_lengths):
        dataset = []
        for chat in data:
            persona_info = [vocab.string2ids(s) for s in chat['persona_info']]
            dialog = [vocab.string2ids(s) for s in chat['dialog']]
            weight = chat['weight']

            if len(dialog) % 2 == 1:
                dialog = dialog[:-1]
                weight = weight[:-1]
           
            # if sum(weight)
            dataset.append((persona_info, dialog, weight))

        return dataset

    def __init__(self, paths, vocab, max_lengths=2048, min_infos=2, int_label=False, persona_enc_policy='link', shuffle_persona=False):
        # policies: 'link' -> B P E B P E; 'concate' -> B P P P E; 'add' -> B P E + B P E; 'sep' -> B P S P S P E
        assert min_infos > 0             

        if isinstance(paths, str):
            paths = [paths]
        
        self.vocab = vocab
        self.max_lengths = max_lengths
        self.min_infos = min_infos
        self.int_label = int_label
        self.persona_enc_policy = persona_enc_policy
        self.shuffle_persona = shuffle_persona

        parsed_data = sum([FacebookDataset.parse_data(path) for path in paths], [])
        if int_label:
            parsed_data = FacebookDataset.parse_int_weight(parsed_data)
        else:
            parsed_data = FacebookDataset.parse_float_weight(parsed_data)
        self.data = FacebookDataset.make_dataset(parsed_data, vocab, max_lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        persona_info, dialog, weight = self.data[idx]

        if len(persona_info):
            if self.persona_enc_policy == 'concate':
                # trick: 随机取persona的sub-set，增强泛化性
                n_info_samples = max(self.min_infos, random.randint(1, len(persona_info)))
                n_info_samples = min(n_info_samples, len(persona_info))
                persona_info = random.sample(persona_info, n_info_samples)
                random.shuffle(persona_info)
                persona_info = sum(persona_info, []) 
                persona_info = [self.vocab.info_bos_id] + persona_info[:self.max_lengths-2] + [self.vocab.info_eos_id]

            elif self.persona_enc_policy == 'link':
                if False: # FIXME: disable
                    # trick: 打乱persona顺序后拼接，增强泛化性
                    seed = random.randint(0,100)
                    random.Random(seed).shuffle(persona_info)
                    for i in weight:
                        if i is not None:
                            random.Random(seed).shuffle(i)
                # random.shuffle(persona_info)
                persona_info = [[self.vocab.info_bos_id] + i + [self.vocab.info_eos_id] for i in persona_info]
                persona_info = sum(persona_info, []) # CONCATE: B P1 E B P2 E ...
                persona_info = persona_info[:self.max_lengths]
                # print(self.max_lengths)
                persona_info = persona_info[:-1] + [self.vocab.info_eos_id]
            
            elif self.persona_enc_policy == 'add':
                seed = random.randint(0,100)
                random.Random(seed).shuffle(persona_info)
                for i in weight:
                    if i is not None:
                        random.Random(seed).shuffle(i)
                persona_info = [[self.vocab.info_bos_id] + i + [self.vocab.info_eos_id] for i in persona_info]
                persona_info = sum(persona_info, []) # CONCATE: B P1 E B P2 E ...
                persona_info = persona_info[:self.max_lengths]
                # print(self.max_lengths)
                persona_info = persona_info[:-1] + [self.vocab.info_eos_id]

            elif self.persona_enc_policy == 'add_pad' or self.persona_enc_policy == 'hier_sep_attn':
                persona_info = [[self.vocab.info_bos_id] + i + [self.vocab.info_eos_id] for i in persona_info]
                # trick: 打乱persona顺序后拼接，增强泛化性
                if self.shuffle_persona:
                    seed = random.randint(0,100)
                    random.Random(seed).shuffle(persona_info)
                    for i in weight:
                        if i is not None:
                            random.Random(seed).shuffle(i)
                # FIXME: max_length limit
                # persona_info = sum(persona_info, []) # CONCATE: B P1 E B P2 E ...
                # persona_info = persona_info[:self.max_lengths]
                # # print(self.max_lengths)
                # persona_info = persona_info[:-1] + [self.vocab.info_eos_id]

            elif self.persona_enc_policy == 'sep':
                pass


            persona_len = []
            align = 0
            for idx, i in enumerate(persona_info):
                if i is self.vocab.info_eos_id:
                    persona_len.append(idx-align-1)
                    align = idx+1

        dialog_begin = 0
        dialog_end = random.randrange(2, len(dialog)+1, 2)

        h = []
        for i, ids in enumerate(dialog[dialog_begin:dialog_end-1], 1):
            if i % 2 == 1:
                ids = [self.vocab.talker1_bos_id] + ids + [self.vocab.talker1_eos_id]
            else:
                ids = [self.vocab.talker2_bos_id] + ids + [self.vocab.talker2_eos_id]
            h.extend(ids)# not append, all history as one sentence
        h = h[-self.max_lengths:]

        y = [self.vocab.bos_id] + dialog[dialog_end-1] + [self.vocab.eos_id]
        y = y[:self.max_lengths]

        w = weight[dialog_end-1]

        return persona_info, h, y, w, persona_len

    
    def convert_single(self, persona_info, query):
        # return persona_info, dialog, persona_len

        persona_info = [self.vocab.string2ids(s) for s in persona_info]
        query = self.vocab.string2ids(query)

        # 用于预测，不做抽样，全部取出
        if self.persona_enc_policy == 'concate':
            persona_info = sum(persona_info, []) 
            persona_info = [self.vocab.info_bos_id] + persona_info[:self.max_lengths-2] + [self.vocab.info_eos_id]

        elif self.persona_enc_policy == 'link':
            persona_info = [[self.vocab.info_bos_id] + i + [self.vocab.info_eos_id] for i in persona_info]
            persona_info = sum(persona_info, []) # CONCATE: B P1 E B P2 E ...
            persona_info = persona_info[:self.max_lengths]
            # print(self.max_lengths)
            persona_info = persona_info[:-1] + [self.vocab.info_eos_id]
        
        elif self.persona_enc_policy == 'add':
            # seed = random.randint(0,100)
            seed = 43
            random.Random(seed).shuffle(persona_info)
            persona_info = [[self.vocab.info_bos_id] + i + [self.vocab.info_eos_id] for i in persona_info]
            persona_info = sum(persona_info, []) # CONCATE: B P1 E B P2 E ...
            persona_info = persona_info[:self.max_lengths]
            # print(self.max_lengths)
            persona_info = persona_info[:-1] + [self.vocab.info_eos_id]

        elif self.persona_enc_policy == 'add_pad':
            persona_info = [[self.vocab.info_bos_id] + i + [self.vocab.info_eos_id] for i in persona_info]

        elif self.persona_enc_policy == 'hier_sep_attn':
            persona_info = [[self.vocab.info_bos_id] + i + [self.vocab.info_eos_id] for i in persona_info]

        elif self.persona_enc_policy == 'sep':
            pass

        persona_len = []
        align = 0
        for idx, i in enumerate(persona_info):
            if i is self.vocab.info_eos_id:
                persona_len.append(idx-align-1)
                align = idx+1

        h = []
        query = [self.vocab.talker2_bos_id] + query + [self.vocab.talker2_eos_id]
        h.extend(query)

        return persona_info, h, persona_len
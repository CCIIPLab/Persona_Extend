import torch
import random
from model.utils import load_openai_weights, set_seed, f1_score, pad_sequence, pad_sequence_of_sequence
from model.transformer_model import TransformerModel
from model.trainer import Trainer
from model.text import BPEVocab
from model.dataset import FacebookDataset
from config import get_model_config, get_trainer_config


def main():
    model_config = get_model_config()
    trainer_config = get_trainer_config()

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)

    transformer = TransformerModel(n_layers=model_config.n_layers,
                                   n_embeddings=len(vocab),
                                   n_pos_embeddings=model_config.n_pos_embeddings,
                                   embeddings_size=model_config.embeddings_size,
                                   padding_idx=vocab.pad_id,
                                   n_heads=model_config.n_heads,
                                   dropout=model_config.dropout,
                                   embed_dropout=model_config.embed_dropout,
                                   attn_dropout=model_config.attn_dropout,
                                   ff_dropout=model_config.ff_dropout,
                                   bos_id=vocab.bos_id,
                                   eos_id=vocab.eos_id,
                                   max_seq_len=model_config.max_seq_len,
                                   beam_size=model_config.beam_size,  
                                   length_penalty=model_config.length_penalty,
                                   n_segments=model_config.n_segments,
                                   annealing_topk=model_config.annealing_topk,
                                   annealing=model_config.annealing,
                                   diversity_coef=model_config.diversity_coef,
                                   diversity_groups=model_config.diversity_groups,
                                   policy=trainer_config['persona_enc_policy'])
    
    print('# generator parameters:', sum(param.numel() for param in transformer.parameters()))

    if not trainer_config.load_last:
        load_openai_weights(transformer.transformer_module, 
                            trainer_config.openai_parameters_dir,
                            n_special_tokens=vocab.n_special_tokens)
        print('OpenAI weights loaded from {}'.format(trainer_config.openai_parameters_dir))

    train_dataset = FacebookDataset(trainer_config.train_datasets, vocab, transformer.n_pos_embeddings - 1,
                                    int_label=trainer_config.int_data,
                                    persona_enc_policy=trainer_config.persona_enc_policy,
                                    shuffle_persona=trainer_config['shuffle_persona'])
    test_dataset = FacebookDataset(trainer_config.test_datasets, vocab, transformer.n_pos_embeddings - 1,
                                    int_label=trainer_config.int_data,
                                    persona_enc_policy=trainer_config.persona_enc_policy)

    model_trainer = Trainer(transformer,
                            train_dataset, 
                            test_dataset, 
                            batch_size=trainer_config.batch_size,
                            batch_split=trainer_config.batch_split, 
                            lr=trainer_config.lr, 
                            lr_warmup=trainer_config.lr_warmup, 
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight, 
                            n_jobs=trainer_config.n_jobs, 
                            clip_grad=trainer_config.clip_grad, 
                            device=device,
                            ignore_idxs=vocab.special_tokens_ids,
                            int_label=trainer_config.int_data,
                            persona_enc_policy=trainer_config.persona_enc_policy,
                            log_name=trainer_config.log_name,
                            freeze_posteria_attn=trainer_config.freeze_posteria_attn
                            )

    if trainer_config.load_last:
        state_dict = torch.load(trainer_config.last_checkpoint_path, map_location=device)
        model_trainer.load_state_dict(state_dict)
        print('Weights loaded from {}'.format(trainer_config.last_checkpoint_path))
    

    # helpers -----------------------------------------------------
    def save_func(epoch):
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path)  

    def sample_text_func(epoch):
        n_samples = 1 if trainer_config.debug else 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, weight, persona_len in samples:
            tensor_persona_info = torch.tensor([persona_info], dtype=torch.long, device=model_trainer.device)
            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            # contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [persona_info, dialog] if len(c) > 0]

            # prediction, out_weights = model_trainer.model.predict_v1(tensor_persona_info, tensor_dialog, persona_len)
            # prediction = model_trainer.model.predict(contexts)[0]
            prediction = model_trainer.model.predict([tensor_persona_info, tensor_dialog])[0]
            
            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            print('\n')
            print('Persona info:\n\t{}'.format(persona_info_str))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    def sample_text_func_v1(epoch):
        n_samples = 1 if trainer_config.debug else 10 # 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [train_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, weight, persona_len in samples:
            tensor_persona_info = torch.tensor([persona_info], dtype=torch.long, device=model_trainer.device)
            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            persona_len = [persona_len]
            
            prediction, out_weights = model_trainer.model.predict_v1(tensor_persona_info, tensor_dialog, persona_len)
            prediction = prediction[0]
            out_weights = [i.cpu().detach().numpy().tolist() for i in out_weights[0]]
            
            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            print('\n')
            print('Persona info:\n\t{}'.format(persona_info_str))
            print('Persona weight target:\n\t{}'.format(weight))
            print('Persona weight:\n\t{}'.format(out_weights))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    def sample_text_func_v2(epoch):
        n_samples = 1 if trainer_config.debug else 10
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, weight, persona_len in samples:
            tensor_persona_info = torch.tensor([persona_info], dtype=torch.long, device=model_trainer.device)
            personas = []
            start_index = 0
            for persona_length in persona_len:
                personas.append(torch.tensor(persona_info[start_index:start_index+persona_length+2], dtype=torch.long, device=model_trainer.device))
                start_index += (persona_length+2)
            persona_emb = [pad_sequence(personas, batch_first=True, padding_value=model_trainer.model.padding_idx, max_len=max(persona_len))]

            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            persona_len = [persona_len]
            
            prediction, out_weights = model_trainer.model.predict_v2(persona_emb, tensor_dialog, persona_len)
            prediction = prediction[0]
            out_weights = [i.cpu().detach().numpy().tolist() for i in out_weights]
            
            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            print('\n')
            print('Persona info:\n\t{}'.format(persona_info_str))
            print('Persona weight target:\n\t{}'.format(weight))
            print('Persona weight:\n\t{}'.format(out_weights))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    def sample_text_func_v3(epoch):
        n_samples = 1 if trainer_config.debug else 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, weight, persona_len in samples:
            p = [[torch.tensor(p, dtype=torch.long, device=model_trainer.device) for p in per] for per in [persona_info]]
            persona_info = pad_sequence_of_sequence(p, batch_first=True, padding_value=model_trainer.model.padding_idx)
            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            
            prediction, out_weights, sig_weights = model_trainer.model.predict_v3(persona_info, tensor_dialog)
            prediction = prediction[0]
            out_weights = [i.cpu().detach().numpy().tolist() for i in out_weights]
            sig_weights = [i.cpu().detach().numpy().tolist() for i in sig_weights]
            
            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            post_prediction, post_out_weights, post_sig_weights = model_trainer.model.predict_v3(persona_info, tensor_dialog, attn='post')
            post_prediction = post_prediction[0]
            post_out_weights = [i.cpu().detach().numpy().tolist() for i in post_out_weights]
            post_sig_weights = [i.cpu().detach().numpy().tolist() for i in post_sig_weights]
            post_prediction_str = vocab.ids2string(post_prediction)

            print('\n --- TEST PRED SAMPLEs:\n')
            print('Persona info:')
            for persona in persona_info[0]:
                print('\t{}'.format(vocab.ids2string(persona.cpu().numpy())))
            print('Persona weight target:\n\t{}'.format(weight))
            print('POST Persona weight (sigmoid):\n\t{}'.format(post_sig_weights))
            print('POST Persona weight (softmax):\n\t{}'.format(post_out_weights))
            print('Persona weight (sigmoid):\n\t{}'.format(sig_weights))
            print('Persona weight (softmax):\n\t{}'.format(out_weights))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('POST Prediction:\n\t{}'.format(post_prediction_str))
            print('Prediction:\n\t{}'.format(prediction_str))
        
        n_samples = 1 if trainer_config.debug else 3
        samples_idxs = random.sample(range(len(train_dataset)), n_samples)
        samples = [train_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, weight, persona_len in samples:
            p = [[torch.tensor(p, dtype=torch.long, device=model_trainer.device) for p in per] for per in [persona_info]]
            persona_info = pad_sequence_of_sequence(p, batch_first=True, padding_value=model_trainer.model.padding_idx)
            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            
            prediction, out_weights, sig_weights = model_trainer.model.predict_v3(persona_info, tensor_dialog)
            prediction = prediction[0]
            out_weights = [i.cpu().detach().numpy().tolist() for i in out_weights]
            sig_weights = [i.cpu().detach().numpy().tolist() for i in sig_weights]
            
            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            post_prediction, post_out_weights, post_sig_weights = model_trainer.model.predict_v3(persona_info, tensor_dialog, attn='post')
            post_prediction = post_prediction[0]
            post_out_weights = [i.cpu().detach().numpy().tolist() for i in post_out_weights]
            post_sig_weights = [i.cpu().detach().numpy().tolist() for i in post_sig_weights]
            post_prediction_str = vocab.ids2string(post_prediction)

            print('\n --- TRAIN PRED SAMPLEs:\n')
            print('Persona info:')
            for persona in persona_info[0]:
                print('\t{}'.format(vocab.ids2string(persona.cpu().numpy())))
            print('Persona weight target:\n\t{}'.format(weight))
            print('POST Persona weight (sigmoid):\n\t{}'.format(post_sig_weights))
            print('POST Persona weight (softmax):\n\t{}'.format(post_out_weights))
            print('Persona weight (sigmoid):\n\t{}'.format(sig_weights))
            print('Persona weight (softmax):\n\t{}'.format(out_weights))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('POST Prediction:\n\t{}'.format(post_prediction_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    
    def sample_text_func_v4(epoch):
        n_samples = 1 if trainer_config.debug else 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, weight, persona_len in samples:
            p = [[torch.tensor(p, dtype=torch.long, device=model_trainer.device) for p in per] for per in [persona_info]]
            persona_info = pad_sequence_of_sequence(p, batch_first=True, padding_value=model_trainer.model.padding_idx)
            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            
            prediction, out_weights = model_trainer.model.predict_v4(persona_info, tensor_dialog)
            prediction = prediction[0]
            out_weights = [i.cpu().detach().numpy().tolist() for i in out_weights]

            prediction_p, out_weights_p = model_trainer.model.predict_v4(persona_info, tensor_dialog, attn='post')
            prediction_p = prediction_p[0]
            out_weights_p = [i.cpu().detach().numpy().tolist() for i in out_weights_p]

            persona_info_str = vocab.ids2string(persona_info[1:-1])

            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])

            prediction_str = vocab.ids2string(prediction)
            prediction_str_p = vocab.ids2string(prediction_p)

            print('\n --- TEST PRED SAMPLEs:\n')
            print('Persona info:')
            for persona in persona_info[0]:
                print('\t{}'.format(vocab.ids2string(persona.cpu().numpy())))
            print('Persona weight target:\n\t{}'.format(weight))
            print('Pri Persona weight:\n\t{}'.format(out_weights))
            print('Post Persona weight:\n\t{}'.format(out_weights_p))

            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Pri Prediction:\n\t{}'.format(prediction_str))
            print('Post Prediction:\n\t{}'.format(prediction_str_p))
        
        n_samples = 1 if trainer_config.debug else 3
        samples_idxs = random.sample(range(len(train_dataset)), n_samples)
        samples = [train_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target, weight, persona_len in samples:
            p = [[torch.tensor(p, dtype=torch.long, device=model_trainer.device) for p in per] for per in [persona_info]]
            persona_info = pad_sequence_of_sequence(p, batch_first=True, padding_value=model_trainer.model.padding_idx)
            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            
            prediction, out_weights = model_trainer.model.predict_v4(persona_info, tensor_dialog)
            prediction = prediction[0]
            out_weights = [i.cpu().detach().numpy().tolist() for i in out_weights]

            prediction_p, out_weights_p = model_trainer.model.predict_v4(persona_info, tensor_dialog, attn='post')
            prediction_p = prediction_p[0]
            out_weights_p = [i.cpu().detach().numpy().tolist() for i in out_weights_p]

            persona_info_str = vocab.ids2string(persona_info[1:-1])

            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])

            prediction_str = vocab.ids2string(prediction)
            prediction_str_p = vocab.ids2string(prediction_p)

            print('\n --- TRAIN PRED SAMPLEs:\n')
            print('Persona info:')
            for persona in persona_info[0]:
                print('\t{}'.format(vocab.ids2string(persona.cpu().numpy())))
            print('Persona weight target:\n\t{}'.format(weight))
            print('Pri Persona weight:\n\t{}'.format(out_weights))
            print('Post Persona weight:\n\t{}'.format(out_weights_p))

            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Pri Prediction:\n\t{}'.format(prediction_str))
            print('Post Prediction:\n\t{}'.format(prediction_str_p))
    

    def single_predict(persona_info, dialog, persona_len):
        if trainer_config.persona_enc_policy == 'concate':
            tensor_persona_info = torch.tensor([persona_info], dtype=torch.long, device=model_trainer.device)
            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            prediction = model_trainer.model.predict([tensor_persona_info, tensor_dialog])[0]
        elif trainer_config.persona_enc_policy == 'add':
            personas = []
            start_index = 0
            for persona_length in persona_len:
                personas.append(torch.tensor(persona_info[start_index:start_index+persona_length+2], dtype=torch.long, device=model_trainer.device))
                start_index += (persona_length+2)
            persona_emb = [pad_sequence(personas, batch_first=True, padding_value=model_trainer.model.padding_idx, max_len=max(persona_len))]

            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            persona_len = [persona_len]
            
            prediction, out_weights = model_trainer.model.predict_v2(persona_emb, tensor_dialog, persona_len)
            prediction = prediction[0]
            out_weights = [i.cpu().detach().numpy().tolist() for i in out_weights]
        elif trainer_config.persona_enc_policy == 'add_pad':
            p = [[torch.tensor(p, dtype=torch.long, device=model_trainer.device) for p in per] for per in [persona_info]]
            persona_info = pad_sequence_of_sequence(p, batch_first=True, padding_value=model_trainer.model.padding_idx)
            tensor_dialog = torch.tensor([dialog], dtype=torch.long, device=model_trainer.device)
            
            prediction, out_weights, sig_weights = model_trainer.model.predict_v3(persona_info, tensor_dialog)
            prediction = prediction[0]
            # out_weights = [i.cpu().detach().numpy().tolist() for i in out_weights]
            
        
        # persona_info_str = vocab.ids2string(persona_info[1:-1])
        # dialog_str = vocab.ids2string(dialog)
        # dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
        # dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
        prediction_str = vocab.ids2string(prediction)

        # print('\n')
        # print('Persona info:\n\t{}'.format(persona_info_str))
        # print('Persona weight:\n\t{}'.format(out_weights))
        # print('Dialog:{}'.format(dialog_str))
        # print('Prediction:\n\t{}'.format(prediction_str))

        return prediction_str

    def original_predict(epoch):
        '''
        {
            "input": "i am ! for my hobby i like to do canning or some whittling .",
            "full_personas": [
                "i like to remodel homes",
                "i like to go hunting",
                "i like to shoot a bow",
                "my favorite holiday is halloween"
            ],
            "label": "i also remodel homes when i am not out bow hunting ."
        }
        '''
        def tsv_write(fp, list):
            for i in list[:-1]:
                fp.write(str(i)+'\t')
            fp.write(str(list[-1])+'\n')

        fp = open(trainer_config.original_result_path, 'w')
        tsv_write(fp, ['persona1', 'persona2', 'persona3', 'persona4', 'persona5'] +
                    ['input', 'output', 'label'] + 
                    ['removed_persona', 'removed_output'] +
                    ['extended_persona', 'extended_output'])

        import json
        with open(trainer_config.original_data, 'r')as f:
            dict_data = json.load(f)

        from tqdm import tqdm
        for item in tqdm(dict_data['data']):
            p = item['full_personas'] # insuf_persona
            m = p[0] # rm_persona
            p.pop(0)
            e = m # ext_persona
            q = item['input'] # query
            r = item['label'] # response
            # predict for 3kinds of persona

            persona_info, history, persona_len = test_dataset.convert_single(p, q)
            persona_info_o, history, persona_len = test_dataset.convert_single(p + [m], q)
            persona_info_e, history, persona_len = test_dataset.convert_single(p + [e], q)
            result_o = single_predict(persona_info_o, history, persona_len)
            # result_r = single_predict(persona_info, history, persona_len)
            # result_e = single_predict(persona_info_e, history, persona_len)

            pads = [''] * (5-1-len(p))
            tsv_write(fp, p + [m] + pads + 
                    [q, result_o, r] +
                    [m, result_o] +
                    [e, result_o]
                    )

        fp.close()

    def inde_predict(epoch):
        '''
        {
            "input": "no , we recently purchased a new house , so we cannot afford it . have you ?",
            "insuf_persona": [
                "i walk three miles every day",
                "i love to spend time with my family",
                "i'm a baby delivery nurse"
            ],
            "rm_persona": "i love disneyland and mickey mouse",
            "ext_persona": "i just bought a house recently.",
            "label": "yes i love mickey mouse such a cute little rat"
        }
        '''
        def tsv_write(fp, list):
            for i in list[:-1]:
                fp.write(str(i)+'\t')
            fp.write(str(list[-1])+'\n')

        with open(trainer_config.inde_result_path, 'w') as fp:
            tsv_write(fp, ['persona1', 'persona2', 'persona3', 'persona4', 'persona5'] +
                    ['input', 'output', 'label'] + 
                    ['removed_persona', 'removed_output'] +
                    ['extended_persona', 'extended_output'])

        import json
        with open(trainer_config.inde_data, 'r')as f:
            dict_data = json.load(f)

        from tqdm import tqdm
        for item in tqdm(dict_data['data']):
            p = item['insuf_persona'] # insuf_persona
            m = item['rm_persona'] # rm_persona
            e = item['ext_persona'] # ext_persona
            q = item['input'] # query
            r = item['label'] # response
            # predict for 3kinds of persona

            persona_info, history, persona_len = test_dataset.convert_single(p, q)
            persona_info_o, history, persona_len = test_dataset.convert_single(p + [m], q)
            persona_info_e, history, persona_len = test_dataset.convert_single(p + [e], q)
            result_o = single_predict(persona_info_o, history, persona_len)
            result_r = single_predict(persona_info, history, persona_len)
            result_e = single_predict(persona_info_e, history, persona_len)

            pads = [''] * (5-1-len(p))

            with open(trainer_config.inde_result_path, 'a') as fp:
                tsv_write(fp, p + [m] + pads + 
                        [q, result_o, r] +
                        [m, result_r] +
                        [e, result_e]
                        )

        fp.close()
        # EVALUATE for 3kinds of generation -> using script
        pass

    def test_func(epoch):
        if (epoch+1) % trainer_config.test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs)
    
    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        return [1-s for s in scores] 

    # helpers -----------------------------------------------------
    

    try:
        # predict
        if trainer_config.inde_result_predict:
            model_trainer.call_funcs_bypass([inde_predict])
        if trainer_config.original_result_predict:
            model_trainer.call_funcs_bypass([original_predict])
        if trainer_config.inde_result_predict or trainer_config.original_result_predict:
            exit()

        # training
        if trainer_config.persona_enc_policy == 'concate':
            model_trainer.call_funcs_bypass([sample_text_func, test_func])
            model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func, sample_text_func, test_func], risk_func=f1_risk)
        elif trainer_config.persona_enc_policy == 'link':
            # model_trainer.call_funcs_bypass([sample_text_func_v1, test_func])
            model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func, test_func, sample_text_func_v1], risk_func=f1_risk)
        elif trainer_config.persona_enc_policy == 'add':
            # model_trainer.call_funcs_bypass([sample_text_func_v2, test_func])
            model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func, test_func, sample_text_func_v2], risk_func=f1_risk) # pure training
        elif trainer_config.persona_enc_policy == 'add_pad':
            # model_trainer.call_funcs_bypass([sample_text_func_v3])
            # model_trainer.call_funcs_bypass([sample_text_func_v3, test_func])
            model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func, test_func, sample_text_func_v3], risk_func=f1_risk) # pure training
            # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[sample_text_func_v3], risk_func=f1_risk) # pure training
            # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func], risk_func=f1_risk) # pure training
        elif trainer_config.persona_enc_policy == 'hier_sep_attn':
            model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func, test_func, sample_text_func_v4], risk_func=f1_risk) # pure training
            # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[], risk_func=f1_risk) # pure training
            # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[sample_text_func_v4], risk_func=f1_risk) # pure training
            model_trainer.call_funcs_bypass([sample_text_func_v4])


    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), trainer_config.interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()


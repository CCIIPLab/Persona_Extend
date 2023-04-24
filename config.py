from attrdict import AttrDict
from model.utils import openai_transformer_config


def get_model_config():
    default_config = openai_transformer_config()
    config = AttrDict({'bpe_vocab_path': './parameters/bpe.vocab',
                       'bpe_codes_path': './parameters/bpe.code',
                       'checkpoint_path': './checkpoints/last_checkpoint', 
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 256,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': None,
                       'annealing': 0,
                       'length_penalty': 0.6,
                       'n_segments': None})

    return config


def get_trainer_config():
    debug = False
    # debug = True # mini dataset & mini predict
    int_data = True # set None for debug
    config = AttrDict({'n_epochs': 100, # 100
                       'batch_size': 264, # (264, 44) for full, (256, 16) for inde
                       'batch_split': 44,
                       'lr': 6.25e-3, # 6.25e-5
                       'lr_warmup': 100, # 16000
                       'lm_weight': 0.9, # 0.5
                       'post_weight': 0.3,
                       'risk_weight': 0,
                       'n_jobs': 16,
                       'label_smoothing': 0.1,
                       'clip_grad': None,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': True, 
                       'openai_parameters_dir': './parameters',
                       'last_checkpoint_path': './checkpoints/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/interrupt_checkpoint',
                       'debug': debug,
                       'int_data': int_data,
                       'persona_enc_policy': 'add_pad', # Optional: concate, link, link, add, sep
                       'log_name': 'default' if not debug else 'debug',
                       'shuffle_persona': True,
                       'freeze_posteria_attn': False, # for ablation study

                        # ----- for predict only -----
                       'inde_result_predict': False,
                       'inde_data': './prm/nliwc_result.json',
                       'inde_result_path': './results/it_convai2_result.tsv',
                     
                       'original_result_predict': False,
                       'original_data': './datasets/ConvAI2/data_original_valid.json',
                       'original_result_path': './results/convai2_result.tsv',
                        # ----------------------------

                       'train_datasets': ['./datasets/debug_dataset/train.txt'] if debug else 
                                        [ './datasets/persona_response_sentencepair_scored/train_self_original_no_cands.txt',
                                        ] if not int_data else 
                                        ['./datasets/persona_response_sentencepair_scored_int/train_self_original_no_cands_int.txt'],
                       'test_datasets': ['./datasets/debug_dataset/train.txt'] if debug else
                                        [ './datasets/persona_response_sentencepair_scored/valid_self_original_no_cands.txt',
                                        ] if not int_data else 
                                        ['./datasets/persona_response_sentencepair_scored_int/valid_self_original_no_cands_int.txt']
                    })

    return config



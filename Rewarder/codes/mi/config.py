from collections import namedtuple


HParams = namedtuple('HParams',
                     'vocab_size, emb_size, hidden_size, layers_num,'
                     'enc_len, dec_len,'
                     'keep_prob, l2_weight, max_gradient_norm, learning_rate,' 
                     'device, batch_size,'
                     'epoches_per_checkpoint, epoches_per_validate, steps_per_train_log,'
                     'sample_num, max_epoch, burn_down, decay_rate,'
                     'vocab_path, ivocab_path, train_data, valid_data, init_emb, model_path'
                     )

hps = HParams(
            vocab_size=-1, # It is to be replaced by true vocabulary size after loading dictionary.
            emb_size=256, hidden_size=800,
            enc_len=41, dec_len=11, layers_num=2,
            keep_prob=0.75, l2_weight=1e-5, 
            max_gradient_norm=2.0, learning_rate=0.001,
            device='/cpu:0', batch_size=256,
            epoches_per_checkpoint=2, epoches_per_validate=1, steps_per_train_log=500,
            max_epoch=24, burn_down=4, decay_rate=0.95,
            sample_num=1, # Generate some poems during training for observation, with greedy search.
            vocab_path="data/mi/vocab.pickle",
            ivocab_path="data/mi/ivocab.pickle",
            train_data="data/mi/train.pickle", # Training data path.
            valid_data="data/mi/valid.pickle", # Validation data path.
            init_emb="", # The path of pre-trained word2vec embedding. Set it to '' if none.
            model_path="model/" # The path to save checkpoints.
        )

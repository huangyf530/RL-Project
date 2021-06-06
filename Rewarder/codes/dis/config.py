from collections import namedtuple


HParams = namedtuple('HParams',
                     'num_classes, batch_size, rnn_num_layers, rnn_cell_size,'
                     'embedding_dims, normalize_embeddings,'
                     'learning_rate, learning_rate_decay_factor,'
                     'max_grad_norm, keep_prob_emb,'
                     'keep_prob_lstm_out, keep_prob_cl_hidden,'
                     'perturb_norm_length,'
                     'num_power_iteration,'
                     'small_constant_for_finite_diff,'
                     'adv_training_method,'
                     'adv_reg_coeff'
                     )

FLAGS = HParams(
            num_classes=3, 
            batch_size=16,
            rnn_num_layers=2,
            rnn_cell_size=512,
            embedding_dims=256,
            normalize_embeddings=True,
            learning_rate=0.001,
            learning_rate_decay_factor=0.9993,
            max_grad_norm=1.0,
            keep_prob_emb=1.0,
            keep_prob_lstm_out=1.0,
            keep_prob_cl_hidden=1.0,

            perturb_norm_length=5.0,
            num_power_iteration=1,
            small_constant_for_finite_diff=1e-1,
            adv_training_method='atvat',
            adv_reg_coeff=1.0
        )

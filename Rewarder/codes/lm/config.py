from collections import namedtuple
HParams = namedtuple('HParams',
                     'vocab_size, hidden_size, emb_size, layer_num,'
                     'batch_size, keep_prob, l2_ratio,'
                     'max_grad_norm, learning_rate, lr_decay,'
                     'device,'
                     'epoches_per_checkpoint,'
                     'epoches_per_validate,'
                     'steps_per_train_log,'
                     'max_epoch, burn_down')

hps = HParams(
    vocab_size=10000, # It is to be replaced by true vocabulary size after loading dictionary
    emb_size=256, hidden_size=512, layer_num=2,
    keep_prob=0.75, l2_ratio=5e-5,
    batch_size=8, max_grad_norm=1.0,
    learning_rate=0.001,
    lr_decay = 0.9,
    device='/cpu:0',
    steps_per_train_log=10,
    epoches_per_checkpoint=1,
    epoches_per_validate=1,
    max_epoch=9,
    burn_down=1
    )

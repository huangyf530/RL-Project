#!/bin/bash
# export CUDA_VISIBLE_DEVICES=5
CONFIG_PATH="pretrained_gpt2/config.json"
MODEL_PATH="pretrained_gpt2/pytorch_model.bin"
VOCAB_PATH="pretrained_gpt2/vocab.txt"

SAVE_PATH=checkpoints/model-style5-luyou
mkdir -p $SAVE_PATH

model_args="--config-path $CONFIG_PATH --model-path $MODEL_PATH --vocab-path $VOCAB_PATH"
train_args="--do-train \
            --batch-size 256 \
            --gradient-accumulation 1 \
            --epoch 3\
            --lr 5e-6"
save_args="--save-path $SAVE_PATH \
           --save-step 50"

log_args="--wandb-name rl-poem-style5-luyou \
          --wandb-project rl-poem \
          --log-steps 1"

srun -G 1 --nodelist=100server -p titan -c 10 --mem 10G --pty python rl_gpt.py $model_args $train_args $save_args $log_args

set +x
#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
CONFIG_PATH="pretrain_models/config.json"
MODEL_PATH="pretrain_models/pytorch.bin"
VOCAB_PATH="pretrain_models/vocab.txt"

model_args="--config-path $CONFIG_PATH --model-path $MODEL_PATH --vocab-path $VOCAB_PATH"
train_args="--do-train \
            --batch-size 128 \
            --gradient-accumulation 1"
log_args="--wandb-name rl-poem \
          --wandb-project rl-poem \
          --log-steps 5"

python rl_gpt.py $model_args $train_args $log_args

set +x
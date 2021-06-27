# RL-Poetry Generation

Use PPO and GPT for poetry generation.

## Train
To train the model, you need first have a pretrained GPT model. The run the following command:
```bash
bash scripts/train.sh
```
The script can be running on slurm machines. You may need remove srun configration when running on normal machines.
```bash
python rl_gpt.py $model_args $train_args $save_args $log_args
```
You can check args meaning in file `rl_gpt.py` or `scripts/train.sh`
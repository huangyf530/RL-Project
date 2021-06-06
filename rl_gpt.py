import os

from Rewarder.Rewarder import Rewarder
from transformers import BertTokenizer, GPT2Config, logging
import torch
from torch.utils.data import DataLoader
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.generate import Generator
from dataset import CustomDataset, collate_fn
import argparse
import logging
import numpy as np
import wandb
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("RLTrain")
parser = argparse.ArgumentParser()

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def init_parser(parser):
    # model args
    parser.add_argument("--config-path", default="./pretrain_models/config.json", type=str)
    parser.add_argument("--model-path", default="./pretrain_models/pytorch.bin", type=str)
    parser.add_argument("--vocab-path", default="./pretrain_models/vocab.txt", type=str)
    
    # train args
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=2021, help="random seed for initialization")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--gradient-accumulation", default=1, type=int)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)

    # data args
    parser.add_argument("--data-path", default="data", type=str)

    # log args
    parser.add_argument("--log-steps", default=1, type=int)
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-name", type=str, default=None)
    # save args
    parser.add_argument("--save-path", default="checkpoints", type=str)
    parser.add_argument("--save-step", default=200, type=int)
    return parser

def model_provider(args):
    # get models
    logger.info("Load model ...")
    model_state_dict = torch.load(args.model_path, map_location='cpu')
    model_config = GPT2Config.from_pretrained(args.config_path)
    gpt2_model = GPT2HeadWithValueModel(model_config)
    gpt2_model_ref = GPT2HeadWithValueModel(model_config)
    gpt2_model.load_state_dict(model_state_dict, strict=False)
    model_state_dict_2 = torch.load(args.model_path, map_location='cpu')
    gpt2_model_ref.load_state_dict(model_state_dict_2, strict=False)
    gpt2_tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    return gpt2_model, gpt2_model_ref, gpt2_tokenizer

def dataset_provider(args, tokenizer, wo_train=False):
    data_path = args.data_path
    logger.info("Load dataset from {:s}".format(data_path))
    valid_dataset = CustomDataset(os.path.join(data_path, "valid.txt"))
    test_dataset = CustomDataset(os.path.join(data_path, "test.txt"))
    if not wo_train:
        train_dataset = CustomDataset(os.path.join(data_path, "train.txt"))
        logger.info("Train {:d} | Valid {:d} | Test {:d}"\
            .format(len(train_dataset), len(valid_dataset), len(test_dataset)))
        return train_dataset, valid_dataset, test_dataset
    else:
        logger.info("Valid {:d} | Test {:d}"\
            .format(len(valid_dataset), len(test_dataset)))
        return valid_dataset, test_dataset

def train(args, model, model_ref, generator, tr_dataset, v_dataset, rewarder):
    # initialize trainer
    ppo_config = {
        'batch_size': args.batch_size * args.gradient_accumulation,
        'forward_batch_size': args.gradient_accumulation,
        'lr': args.lr}
    ppo_trainer = PPOTrainer(model, model_ref, **ppo_config)
    # initialize dataloader
    tr_dataloader = DataLoader(tr_dataset, args.batch_size, shuffle=True, \
        collate_fn=collate_fn, num_workers=4)
    logger.info("*" * 10 + "Begin Training" + '*' * 8)
    logger.info(f"batch size = {args.batch_size}")
    logger.info(f"Epoch num  = {args.epoch}")
    logger.info(f"Block num  = {len(tr_dataloader)}")
    if args.wandb_name is not None:
        wandb.init(name=args.wandb_name, project=args.wandb_project)
        wandb.config.update(args)
    total_step = 0
    logs = {'env/reward_mean': [], 'env/reward_std': [], 'env/reward_dist': [],
            'env/lm_mean': [], 'env/mi_mean': [], 'env/tfidf_mean': [], 'env/quality_mean': []}
    timing = {'time/get_response': [],
              'time/get_rewards': [],
              'time/optimization': [],
              'time/step': []}
    model.train()
    for epoch in range(args.epoch):
        for step, data in enumerate(tr_dataloader):
            t0 = time.time()
            # get model response
            genres, keywords, poems = data
            t = time.time()
            total_tokens, query_tensor, response_tensor = generator.generate(keywords, genres)
            generated_poems = []
            for poem in total_tokens:
                generated_poems.append(poem.split('#')[-1])
            timing['time/get_response'].append(time.time()-t)
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            t = time.time()
            final_scores, lm_vals, mi_vals, tfidf_vals, quality_vals = rewarder.get_mixed_scores_with_poems(poems, fixed_sens_num=4)
            reward_tensor = torch.from_numpy(final_scores).to(model.device)
            timing['time/get_rewards'].append(time.time()-t)
            # train model with ppo
            t = time.time()
            train_stats = ppo_trainer.step(query_tensor, response_tensor, reward_tensor)
            timing['time/optimization'].append(time.time()-t)
            timing['time/step'].append(time.time() - t0)
            logs.update(train_stats)
            logs['env/reward_mean'].append(torch.mean(reward_tensor).cpu().numpy())
            logs['env/reward_std'].append(torch.std(reward_tensor).cpu().numpy())
            logs['env/reward_dist'].append(reward_tensor.cpu().numpy())
            logs['env/lm_mean'].append(np.mean(lm_vals))
            logs['env/mi_mean'].append(np.mean(mi_vals))
            logs['env/tfidf_mean'].append(np.mean(tfidf_vals))
            logs['env/quality_mean'].append(np.mean(quality_vals))
            total_step += 1
            if total_step % args.log_steps == 0:
                logs['env/reward_mean'] = np.mean(logs['env/reward_mean'])
                logs['env/reward_std'] = np.mean(logs['env/reward_std'])
                logs['env/reward_dist'] = np.mean(logs['env/reward_dist'], axis=0)
                logs['env/lm_mean'] = np.mean(logs['env/lm_mean'])
                logs['env/mi_mean'] = np.mean(logs['env/mi_mean'])
                logs['env/tfidf_mean'] = np.mean(logs['env/tfidf_mean'])
                logs['env/quality_mean'] = np.mean(logs['env/quality_mean'])
                for key in timing:
                    timing[key] = np.mean(timing[key])
                logger.info("Step {:d} | reward {:.5f} | lm {:.5f} | mi {:.5f} | tfidf {:.5f} | quality {:.5f} | per step {:.3f}".format(total_step, logs['env/reward_mean'], logs['env/lm_mean'], logs['env/mi_mean'], logs['env/tfidf_mean'], logs['env/quality_mean'], timing['time/step']))
                logs.update(timing)
                if args.wandb_name is not None:
                    wandb.log(logs)
                for key in timing:
                    timing[key] = []
                logs['env/reward_mean'] = []
                logs['env/reward_std'] = []
                logs['env/reward_dist'] = []
                logs['env/lm_mean'] = []
                logs['env/mi_mean'] = []
                logs['env/tfidf_mean'] = []
                logs['env/quality_mean'] = []
            if total_step % args.save_step == 0:
                save_path = os.path.join(args.save_path, f'epoch-{epoch}-step-{step}.pt') 
                torch.save(generator.model.state_dict(), save_path)
                logger.info("Epoch {:d} | Total step {:d} | Save model to {:s}"\
                    .format(epoch, total_step, save_path))
        save_path = os.path.join(args.save_path, f'epoch-{epoch}.pt')
        torch.save(generator.model.state_dict(), save_path)
        logger.info("Epoch {:d} | Save model to {:s}".format(epoch, save_path))

if __name__=="__main__":
    parser = init_parser(parser)
    args = parser.parse_args()
    logger.info(args)
    # device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tf_device = '/GPU:0' if torch.cuda.is_available() and not args.no_cuda else '/cpu:0'
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    logger.warning(f"device: {device}, n_gpu: {args.n_gpu}")
    set_seed(args)
    # rewarder
    rewarder = Rewarder(sep_device=tf_device)
    
    # model
    gpt2_model, gpt2_model_ref, gpt2_tokenizer = model_provider(args)
    gpt2_model.to(device)
    gpt2_model_ref.to(device)
    generator = Generator(gpt2_model, gpt2_tokenizer, args.do_train)

    if args.do_train:
        tr_dataset, v_dataset, te_dataset = dataset_provider(args, gpt2_tokenizer)
    else:
        v_dataset, te_dataset = dataset_provider(args, gpt2_tokenizer, True)
    
    if args.do_train:
        train(args, gpt2_model, gpt2_model_ref, generator, tr_dataset, v_dataset, rewarder)
    # keywords = [['春天', '夏天', '秋天', '冬天']] * 2
    # genres = [1, 2]
    # print(generator.generate(keywords, genres))
    # generate(gpt2_model, keywords, genre, gpt2_tokenizer)
    
    # rewarder = Rewarder()
    # mu, sigma = -4.606461+0.5, 1.333839
    # poem = "春眠不觉晓|处处闻啼鸟|夜来风雨声|花落知多少"
    # print("Reward:", rewarder.get_poem_scores(poem))
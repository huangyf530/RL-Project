import os

from Rewarder.Rewarder import Rewarder
from PoetReward import Rewarder as PoetRewarder
from get_ppl_conditional import Generator as LMRewarder
from transformers import BertTokenizer, GPT2Config, logging
import torch
from torch.utils.data import DataLoader
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.generate3 import Generator
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

def train(args, model, model_ref, generator, tr_dataset, v_dataset, poet_rewarder, lm_rewarder=None):
    # initialize trainer
    ppo_config = {
        'batch_size': args.batch_size * args.gradient_accumulation,
        'forward_batch_size': int(args.batch_size / 8),
        'lr': args.lr,
        'init_kl_coef': 0.3}
    ppo_trainer = PPOTrainer(model, model_ref, **ppo_config)
    # initialize dataloader
    tr_dataloader = DataLoader(tr_dataset, args.batch_size, shuffle=True, \
        collate_fn=collate_fn, num_workers=4)
    logger.info("*" * 10 + " Begin Training " + '*' * 10)
    logger.info(f"    batch size = {args.batch_size}")
    logger.info(f"    gradient accumulation = {args.gradient_accumulation}")
    logger.info(f"    Epoch num  = {args.epoch}")
    logger.info(f"    Block num  = {len(tr_dataloader)}")
    logger.info(f"    Total step = {len(tr_dataloader) * args.epoch}")
    if args.wandb_name is not None:
        wandb.init(name=args.wandb_name, project=args.wandb_project, settings=wandb.Settings(start_method="fork"))
        wandb.config.update(args)
    total_step = 0
    # to_log = {'reward': ['mean', 'std', 'dist'], 'mi': ['mean'], 'tfidf': ['mean'], 'lm': ['mean'], 'quality': ['mean']}
    to_log = {'reward': ['mean', 'std', 'dist'], "poet_reward": ['mean'], "lm_reward": ['mean']}
    logs = {}
    for key in to_log:
        for metric in to_log[key]:
            logs[f"env/{key}_{metric}"] = []
    # timing = {'time/get_response': [],
    #           'time/get_rewards': [],
    #           'time/optimization': [],
    #           'time/step': []}
    alpha = [0, 2, 2, 0]
    beta = [0, 0.5, 0.5, 0]
    model.train()
    if args.wandb_name is not None:
        wandb.watch(model, log='all')
    reward_tensors_list = []
    query_tensors_list = []
    response_tensors_list = []
    for epoch in range(args.epoch):
        for step, data in enumerate(tr_dataloader):
            t0 = time.time()
            # get model response
            genres, keywords, poems = data
            t = time.time()
            total_tokens, query_tensor, response_tensor = generator.generate(keywords, genres)
            generated_poems = []
            print(generator.tokenizer.decode(query_tensor[1,:]))
            print(generator.tokenizer.decode(response_tensor[1, :]))
            print(total_tokens[1][5:])
            # quit()
            for poem in total_tokens:
                generated_poems.append(poem.split('|')[-1])
            # timing['time/get_response'].append(time.time()-t)
            # define a reward for response
            # (this could be any reward such as human feedback or output from another model)
            t = time.time()
            # final_scores, lm_vals, mi_vals, tfidf_vals, quality_vals = rewarder.\
            #     get_mixed_scores_with_poems(generated_poems, fixed_sens_num=4)
            # # modify the range of reward
            # lm_vals = np.clip(lm_vals, a_min=beta[0], a_max=None) - beta[0]
            # mi_vals = np.clip(mi_vals, a_min=beta[1], a_max=None) - beta[1]
            # tfidf_vals = np.clip(tfidf_vals, a_min=beta[2], a_max=None) - beta[2]
            # quality_vals = np.clip(quality_vals, a_min=beta[3], a_max=None) - beta[3]
            # final_scores = alpha[0] * lm_vals \
            #          + alpha[1] * mi_vals \
            #          + alpha[2] * tfidf_vals \
            #          + alpha[3] * quality_vals
            # reward_dict = {'reward': final_scores, "lm": lm_vals, "mi": mi_vals, "tfidf": tfidf_vals, 'quality': quality_vals}
            conditional_poems = []
            for tokens in total_tokens:
                conditional_poems.append(tokens[5:])
            poet_scores = poet_rewarder.reward(generated_poems)
            # lm_scores = lm_rewarder.get_scores(conditional_poems)
            poet_scores = np.array(poet_scores)
            # lm_scores = np.array(lm_scores)
            poet_scores *= 5
            # lm_scores *= 2
            final_scores = poet_scores
            reward_dict = {'reward': final_scores, 'poet_reward': poet_scores, "lm_reward": lm_scores}
            for key in to_log:
                r = reward_dict[key]
                for metric in to_log[key]:
                    if metric == 'mean':
                        logs[f'env/{key}_{metric}'].append(np.mean(r))
                    elif metric == 'std':
                        logs[f'env/{key}_{metric}'].append(np.std(r))
                    elif metric == 'dist':
                        logs[f'env/{key}_{metric}'].append(r)
                    else:
                        raise ValueError(f"{metric} is not legal, should be one of [mean, std, dist]")
            # mu = np.mean(final_scores)
            # std = np.std(final_scores)
            # final_scores = (final_scores - mu) / (std + 1e-7)
            reward_tensor = torch.from_numpy(final_scores).to(model.device)
            reward_tensors_list.append(reward_tensor)
            query_tensors_list.append(query_tensor)
            response_tensors_list.append(response_tensor)
            # timing['time/get_rewards'].append(time.time()-t)
            if (step + 1) % args.gradient_accumulation == 0:
                # train model with ppo
                # print(reward_tensors_list)
                # print(query_tensors_list)
                # print(response_tensors_list)
                reward_tensor = torch.cat(reward_tensors_list, 0)
                query_tensor = torch.cat(query_tensors_list, 0)
                response_tensor = torch.cat(response_tensors_list, 0)
                reward_tensors_list = []
                query_tensors_list = []
                response_tensors_list = []
                t = time.time()
                train_stats = ppo_trainer.step(query_tensor, response_tensor, reward_tensor)
                # timing['time/optimization'].append(time.time()-t)
                # timing['time/step'].append(time.time() - t0)
                logs.update(train_stats)
                total_step += 1
                if total_step % args.log_steps == 0:
                    for key in to_log:
                        for metric in to_log[key]:
                            if metric == 'mean' or metric == 'std':
                                logs[f'env/{key}_{metric}'] = np.mean(logs[f'env/{key}_{metric}'])
                            elif metric == 'dist':
                                logs[f'env/{key}_{metric}'] = np.mean(logs[f'env/{key}_{metric}'], axis=0)
                            else:
                                raise ValueError(f"{metric} is not legal, should be one of [mean, std, dist]")
                    # # log mean timing
                    # for key in timing:
                    #     timing[key] = np.mean(timing[key])
                    loginfo = "Step {:d}".format(total_step)
                    # log mean
                    for key in to_log:
                        loginfo += " | {:s} {:.5f}".format(key, logs[f'env/{key}_mean'])
                    # loginfo += " | per step {:.3f}s".format(timing['time/step'])
                    logger.info(loginfo)
                    # logs.update(timing)
                    if args.wandb_name is not None:
                        wandb.log(logs, step=total_step)
                    # reset
                    # for key in timing:
                    #     timing[key] = []
                    for key in to_log:
                        for metric in to_log[key]:
                            logs[f"env/{key}_{metric}"] = []
                if total_step % args.save_step == 0:
                    save_path = os.path.join(args.save_path, f'epoch-{epoch}-step-{total_step}.pt') 
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
    # rewarder = Rewarder(sep_device=tf_device)
    poet_rewarder = PoetRewarder('/home/liwenhao/bert_new/models/Poets/陆游', '/home/liwenhao/bert_new/models/v1', 0)
    # lm_rewarder = LMRewarder('/home/liwenhao/GPT2_new/models/gpt_v1/checkpoint-59140')
    
    # model
    gpt2_model, gpt2_model_ref, gpt2_tokenizer = model_provider(args)
    gpt2_model.to(device)
    gpt2_model_ref.to(device)
    generator = Generator(gpt2_model, gpt2_tokenizer, args.do_train)
    logger.info(gpt2_model)

    if args.do_train:
        tr_dataset, v_dataset, te_dataset = dataset_provider(args, gpt2_tokenizer)
    else:
        v_dataset, te_dataset = dataset_provider(args, gpt2_tokenizer, True)
    
    if args.do_train:
        train(args, gpt2_model, gpt2_model_ref, generator, tr_dataset, v_dataset, poet_rewarder, lm_rewarder)
    # keywords = [['春天', '夏天', '秋天', '冬天']] * 2
    # genres = [1, 2]
    # print(generator.generate(keywords, genres))
    # generate(gpt2_model, keywords, genre, gpt2_tokenizer)
    
    # rewarder = Rewarder()
    # mu, sigma = -4.606461+0.5, 1.333839
    # poem = "春眠不觉晓|处处闻啼鸟|夜来风雨声|花落知多少"
    # print("Reward:", rewarder.get_poem_scores(poem))
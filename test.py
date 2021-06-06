from Rewarder.Rewarder import Rewarder
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer

def model_provider(pretrain_name_or_path):
    # get models
    gpt2_model = GPT2HeadWithValueModel.from_pretrained(pretrain_name_or_path)
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(pretrain_name_or_path)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(pretrain_name_or_path)

def train(args, model, model_ref):
    # initialize trainer
    ppo_config = {'batch_size': args.batch_size, 'forward_batch_size': args.forward_batch_size}
    ppo_trainer = PPOTrainer(model, model_ref, **ppo_config)

if __name__=="__main__":
    rewarder = Rewarder()
    mu, sigma = -4.606461+0.5, 1.333839
    poem = "春眠不觉晓|处处闻啼鸟|夜来风雨声|花落知多少"
    print("Reward:", rewarder.get_poem_scores(poem))

from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
import torch
import re
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import random
from trl.inf_planner import BatchMasker
random.seed(952)


class Generator(object):

    genre_dict = [None, '《', '》']
    yan_dic = [-1, 5, 7]

    def __init__(self, config_path, ckpt_path=None):
        if ckpt_path is None:
            self.model = GPT2LMHeadModel.from_pretrained(config_path)
        else:
            model_config = GPT2Config.from_pretrained(config_path)
            model_state_dict = torch.load(ckpt_path)
            del model_state_dict['v_head.summary.weight']
            del model_state_dict['v_head.summary.bias']
            self.model = GPT2LMHeadModel(model_config)
            self.model.load_state_dict(model_state_dict)
        self.model = self.model.cuda()
        self.tokenizer = BertTokenizer.from_pretrained(config_path)
        self.top_k = 15
        self.pad_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.model.eval()

    def next_prob(self, seq, token):
        with torch.no_grad():
            inputs = {'input_ids': seq}
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            logits = F.softmax(next_token_logits, dim=-1)
            return torch.gather(logits, dim=1, index=token).reshape(-1).cpu().detach().numpy()

    def get_ppl(self, seqs):
        prefix_length = 15
        max_length = 50
        seq_lengths = [len(seq) + 2 for seq in seqs]
        ret = []
        for prefix in seqs:
            ret.append(self.tokenizer(prefix, padding='max_length', max_length=50)['input_ids'])
            # print(ret[-1], ret[-1][prefix_length - 1])
            assert ret[-1][prefix_length - 1] == 6687
        ppls = [0 for _ in seqs]
        ret = torch.LongTensor(ret).cuda()
        for i in range(prefix_length, max_length):
            res = self.next_prob(ret[:, :i], ret[:, i].reshape(-1, 1))
            for j in range(len(seqs)):
                if i < seq_lengths[j]:
                    ppls[j] += np.log(res[j])
        for j in range(len(seqs)):
            ppls[j] /= (seq_lengths[j] - prefix_length)
        return ppls

    def generate_file(self, infile, outfile, batch_size):
        lines = []
        fin = open(infile)
        fout = open(outfile, 'w')
        all_ppls = []
        for line in tqdm(fin.readlines()):
            lines.append(line.strip())
            if len(lines) % batch_size == 0:
                ppls = self.get_ppl(lines)
                all_ppls.extend(ppls)
                lines = []
        ppls = self.get_ppl(lines)
        all_ppls.extend(ppls)
        print(np.mean(all_ppls), np.var(all_ppls))

    def get_scores(self, sens):
        mu = -3.9565833552412424
        sigma = 0.6026663561416339
        ppls = self.get_ppl(sens)
        scores = [np.exp(-max(np.abs(v - mu) - 0.25 * sigma, 0.0)) for v in ppls]
        return scores

if __name__=="__main__":
    model_path = '/home/liwenhao/GPT2_new/models/gpt_v1/checkpoint-59140'

    generator = Generator(model_path)
    sens = ['《$秋风；疏桐；饮#；清露|垂緌饮清露，流响出疏桐，居高声自远，非是藉秋风'] * 3
    genres = [1, 1, 2]
    #print(generator.generate_file('data/quatrain_research_valid.txt', 'outs/test.txt', 16))
    print(generator.get_scores(sens))
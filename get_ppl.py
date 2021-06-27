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
            next_token_logits = outputs[0][:, -1, :]
            logits = F.softmax(next_token_logits, dim=-1)
            return torch.gather(logits, dim=1, index=token).reshape(-1).cpu().detach().numpy()

    def get_ppl(self, seqs):
        seq_lengths = [len(seq) + 1 for seq in seqs]
        ret = []
        for prefix in seqs:
            ret.append(self.tokenizer(prefix, padding='max_length', max_length=9)['input_ids'])
        ppls = [0 for _ in seqs]
        ret = torch.LongTensor(ret).cuda()
        for i in range(8):
            res = self.next_prob(ret[:, :(i + 1)], ret[:, i + 1].reshape(1, -1))
            for j in range(len(seqs)):
                if i < seq_lengths[j]:
                    ppls[j] += np.log(res[j])
        for j in range(len(seqs)):
            ppls[j] /= seq_lengths[j]
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
        mu = -8.462329023165363
        sigma = 3.5936490517485953
        all_sens = []
        for sen in sens:
            sen = sen.strip().split('，')
            assert len(sen) == 4
            all_sens.extend(sen)
        ppls = self.get_ppl(all_sens)
        scores = [0 for _ in sens]
        for i in range(len(sens)):
            for j in range(4):
                v = ppls[i * 4 + j]
                a = max(np.abs(v - mu) - 0.25 * sigma, 0.0)
                a = np.exp(-a)
                scores[i] += a
        scores = [ele / 4 for ele in scores]
        return scores




if __name__=="__main__":
    model_path = '/home/liwenhao/GPT2_new/models/gpt_os2/checkpoint-59130'

    generator = Generator(model_path)
    sens = ['床前明月光，疑是地上霜，举头望明月，低头思故乡'] * 3
    genres = [1, 1, 2]
    print(generator.get_scores(sens))
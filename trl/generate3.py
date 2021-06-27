from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
import torch
import re
import torch.nn.functional as F
from tqdm import tqdm
import random
from .inf_planner import BatchMasker
random.seed(952)


class Generator(object):

    genre_dict = [None, '《', '》']
    yan_dic = [-1, 5, 7]

    def __init__(self, model, tokenizer, training=False):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = 15
        self.pad_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.training = training
        print('load finish')

    def generate(self, keywords, genres):
        prefixes = self.encode_prefix(keywords, genres)
        prefixes = self.tokenize_prefixes(prefixes)
        return self.sample_sequence(prefixes, genres)

    def encode_prefix(self, keywords, genres):
        prefixes = []
        for keyword, genre in zip(keywords, genres):
            kws = []
            for ele in keyword:
                assert len(ele) <= 2
                if len(ele) == 1:
                    kws.append(ele + '#')
                elif len(ele) == 2:
                    kws.append(ele)
            assert len(kws) <= 4
            while len(kws) < 4:
                kws.append('##')
            kws = '；'.join(kws)
            prefix = self.genre_dict[genre] + '$' + kws + '|'
            prefixes.append(prefix)
        return prefixes

    def tokenize_prefixes(self, prefixes):
        ret = []
        for prefix in prefixes:
            ret.append(self.tokenizer(prefix)['input_ids'][:-1])
        return ret


    def filtering(self, logits, mask, filter_value=-float('Inf')):
        if self.top_k > 0:
            filter_tensor = torch.ones_like(logits) * filter_value
            logits = torch.where(mask, logits, filter_tensor)
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_step(self, seq, mask):
        inputs = {'input_ids': seq}
        outputs = self.model(**inputs)
        next_token_logits = outputs[0][:, -1, :]
        filtered_logits = self.filtering(next_token_logits, mask)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        seq = torch.cat((seq, next_token), dim=1)
        return seq, next_token

    def sample_sequence(self, seq, genres):
        query_tensor = None
        with torch.no_grad():
            masker = BatchMasker([self.yan_dic[i] for i in genres], [random.randint(0, 3) for _ in genres], self.tokenizer, self.model.config.vocab_size, self.model.device)
            all_preds = torch.LongTensor(seq).to(self.model.device)
            query_tensor = all_preds
            for i in range(32):
                mask = masker.step()
                all_preds, nxt = self.sample_step(all_preds, mask)
                masker.update_tokens(nxt)
        query_len = query_tensor.shape[1]
        response_tensor = all_preds[:, query_len:]
        all_preds = all_preds.cpu().tolist()
        pad_id = self.pad_id
        all_preds = [e[:e.index(pad_id)] if pad_id in e else e for e in all_preds]
        return [self.tokenizer.decode(line) for line in all_preds], query_tensor, response_tensor

    def generate_file(self, infile, outfile, batch_size):
        prefixes, genres = [], []
        fin = open(infile)
        fout = open(outfile, 'w')
        for lines in tqdm(fin.readlines()):
            assert lines[0] in self.genre_dict
            genres.append(self.genre_dict.index(lines[0]))
            prefixes.append(lines.split('|')[0] + '|')
            if len(genres) % batch_size == 0:
                seq = self.tokenize_prefixes(prefixes)
                res = self.sample_sequence(seq, genres)
                for ele in res:
                    fout.write(ele.replace(' ', '') + '\n')
                prefixes, genres = [], []
        seq = self.tokenize_prefixes(prefixes)
        res = self.sample_sequence(seq, genres)
        for ele in res:
            fout.write(ele.replace(' ', '') + '\n')


if __name__=="__main__":
    model_path = '/home/liwenhao/GPT2_new/models/gpt_v1/checkpoint-59140'

    generator = Generator(model_path)
    keywords = [['第', '耕田', '识', '好事'], ['你', '你好'], ['春天', '夏天', '秋天', '冬天']]
    genres = [1, 1, 2]
    print(generator.generate_file('data/quatrain_research_valid.txt', 'outs/gpt_v1_valid2.txt', 256))
from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
import torch
import re
import torch.nn.functional as F
from tqdm import tqdm

class GLController:

    yan_dic = [-1, 5, 7]
    line_dic = [-1, 4, 4]

    def __init__(self, genre):
        self.yan = self.yan_dic[genre] + 1
        self.line = self.line_dic[genre]
        self.i = 0

    def step(self):
        self.i += 1
        if self.i >= self.line * self.yan:
            return 'pad'
        if self.i % self.yan == 0:
            return 'sep'
        return 'char'

class MaskGenerator:

    def __init__(self, tokenizer, vocab_size, genres, device):
        self.vocab_size = vocab_size
        self.device = device
        self.tokenizer = tokenizer
        self.masks = self.get_masks(tokenizer)
        self.controllers = [GLController(genre) for genre in genres]

    def lis2mask(self, lis):
        ret = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
        ret[lis] = False
        return ret

    def lis2occ(self, lis):
        ret = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        ret[lis] = True
        return ret

    def mask_combine(self, masks):
        ret = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
        for mask in masks:
            ret = ret & mask
        return ret

    def get_masks(self, tokenizer):
        return {
            'sep': self.lis2occ(tokenizer.convert_tokens_to_ids(['，'])),
            'pad': self.lis2occ(tokenizer.convert_tokens_to_ids(['[PAD]'])),
            'char': self.lis2mask(tokenizer.convert_tokens_to_ids(['《', '》', '；', '#', '$', '|', '[PAD]', '[UNK]', '[CLS]', '[SEP]'])),
        }

    def step(self, seq):
        ret = []
        for i in range(len(self.controllers)):
            mask_type = self.controllers[i].step()
            mask = self.masks[mask_type]
            if mask_type is 'char':
                mask = self.mask_combine([mask, self.lis2mask(seq[i])])
            ret.append(mask.unsqueeze(0))
        return torch.cat(ret, dim=0)


class Generator(object):

    genre_dict = [None, '《', '》']

    def __init__(self, model, tokenizer, training=False):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = 15
        self.training = training
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')

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
        return seq

    def sample_sequence(self, seq, genres):
        query_tensor = None
        if not self.training:
            with torch.no_grad():
                masker = MaskGenerator(self.tokenizer, self.model.config.vocab_size, genres, self.model.device)
                all_preds = torch.LongTensor(seq).to(self.model.device)
                query_tensor = all_preds
                for i in range(32):
                    mask = masker.step(all_preds)
                    all_preds = self.sample_step(all_preds, mask)
        else:
            masker = MaskGenerator(self.tokenizer, self.model.config.vocab_size, genres, self.model.device)
            all_preds = torch.LongTensor(seq).to(self.model.device)
            query_tensor = all_preds
            for i in range(32):
                mask = masker.step(all_preds)
                all_preds = self.sample_step(all_preds, mask)
        query_len = query_tensor.shape[1]
        response_tensor = all_preds[:, query_len:-1]
        all_preds = all_preds.cpu().tolist()
        pad_id = self.pad_id
        all_preds = [e[:e.index(pad_id)] if pad_id in e else e for e in all_preds]
        return [self.tokenizer.decode(line) for line in all_preds], query_tensor, response_tensor

if __name__=="__main__":
    model_path = '/home/liwenhao/GPT2_new/models/gpt_v1/checkpoint-23656'

    generator = Generator(model_path)
    keywords = [['第', '耕田', '识', '好事'], ['你', '你好'], ['春天', '夏天', '秋天', '冬天']]
    genres = [1, 1, 2]
    print(generator.generate(keywords, genres))
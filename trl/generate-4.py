from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
import torch
import re
import torch.nn.functional as F

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
            'char': self.lis2mask(tokenizer.convert_tokens_to_ids(['[PAD]', '[CLS]', '[SEP]', '[UNK]', '#', '|', '，', '1', '2'])),
        }

    def step(self, seq):
        ret = []
        for i in range(len(self.controllers)):
            mask_type = self.controllers[i].step()
            mask = self.masks[mask_type]
            if mask_type is 'char':
                mask = self.mask_combine([mask, self.lis2mask(self.tokenizer.encode(seq[i].split('#')[-1]))])
            ret.append(mask.unsqueeze(0))
        return torch.cat(ret, dim=0)


class Generator(object):
    def __init__(self, model, tokenizer, training=False):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = 15
        self.training = training
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        # self.model.eval()
        # print('load finish')

    def generate(self, keywords, genres):
        prefixes = self.encode_prefix(keywords, genres)
        return self.sample_sequence(prefixes, genres)

    def gather_batch(self, seqences):
        all_dic = {key : [] for key in seqences[0].keys()}
        for inp in seqences:
            for key, ele in inp.items():
                all_dic[key].append(ele)

    def encode_prefix(self, keywords, genres):
        prefixes = []
        for keyword, genre in zip(keywords, genres):
            keyword = '|'.join(keyword)
            prefixes.append('#'.join([str(genre), keyword, ""]))
        return prefixes


    def tokenize_seq(self, seq):
        max_length = 50
        lengths = []
        # ret_dict = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}
        tokenized_prefix = self.tokenizer(seq, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        for i, line in enumerate(seq):
            l = len(re.sub('\[[^\]]*?\]', '@', line))
            tokenized_prefix['input_ids'][i][l + 1] = self.pad_id
            tokenized_prefix['attention_mask'][i][l + 1] = 0
            lengths.append([l])
        print(tokenized_prefix)
        quit()
        return tokenized_prefix, torch.LongTensor(lengths)

    def filtering(self, logits, mask, filter_value=-float('Inf')):
        if self.top_k > 0:
            filter_tensor = torch.ones_like(logits) * filter_value
            logits = torch.where(mask, logits, filter_tensor)
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_step(self, seq, mask):
        inputs, lengths = self.tokenize_seq(seq)
        outputs = self.model(**inputs)
        lengths = lengths.unsqueeze(-1).repeat(1, 1, outputs[0].shape[2])
        # print(outputs[0].shape, lengths.shape)
        next_token_logits = torch.gather(outputs[0], 1, lengths)
        next_token_logits = next_token_logits.squeeze(1)
        # print(next_token_logits.shape)
        filtered_logits = self.filtering(next_token_logits, mask)
        # print(filtered_logits.shape, mask.shape)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).reshape(-1).cpu().tolist()
        # print(next_token)
        next_token = self.tokenizer.convert_ids_to_tokens(next_token)
        # print(next_token)
        for i in range(len(seq)):
            seq[i] += next_token[i]
        return seq

    def sample_sequence(self, seq, genres):
        query_tensor = None
        with torch.no_grad():
            masker = MaskGenerator(self.tokenizer, self.model.config.vocab_size, genres, self.model.device)
            for i in range(32):
                mask = masker.step(seq)
                seq = self.sample_step(seq, mask)
        # all_preds = [[ele for ele in seq if ele != self.pad_id] for seq in all_preds]
        seq = [re.sub('\[[^\]]*?\]', '', line) for line in seq]
        return seq

if __name__=="__main__":
    model_path = '/data/disk2/private/hujinyi/poem_vae/model_no_tag_in_decoder/model_gpt2_keywords/model_epoch_10.pt'
    model_config_path = '/data/disk2/private/hujinyi/poem_vae/data/config.json'
    vocab_path = '/data/disk2/private/hujinyi/poem_vae/data/vocab.txt'

    generator = Generator(model_path, model_config_path, vocab_path)
    keywords = [['第', '耕田', '识', '好事'], ['你', '你好'], ['春天', '夏天', '秋天', '冬天']]
    genres = [1, 1, 2]
    print(generator.generate(keywords, genres))
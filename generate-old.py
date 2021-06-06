from transformers import GPT2LMHeadModel, GPT2Config, BertTokenizer
import torch
import torch.nn.functional as F
class Generator(object):
    def __init__(self, model_path, model_config_path, vocab_path):
        model_state_dict = torch.load(model_path, map_location='cpu')
        model_config = GPT2Config.from_pretrained(model_config_path)
        self.model = GPT2LMHeadModel(model_config)
        self.model.load_state_dict(model_state_dict)
        self.tokenizer = BertTokenizer(vocab_file=vocab_path)
        self.top_k=15
        print('load finish')

    def generate(self, keywords, genre):
        genre = genre = '1' if genre == '五言绝句' else '2'
        keywords = '|'.join(keywords)
        prefix = '#'.join([genre, keywords, ""])
        print(prefix)
        tokens = self.tokenizer(prefix)
        self.generated = torch.LongTensor(tokens['input_ids'][:-1]).unsqueeze(0)
        output = self.sample_sequence()
        print(self.tokenizer.decode(output.tolist()[0]))
    def filtering(self, logits, filter_value=-float('Inf')):
        assert logits.dim() == 1
        if self.top_k > 0:
            generated = self.generated[0].tolist()
            generated = [index for index in generated if index != 0 and index != 6]
            for index in generated:
                logits[index] = filter_value
            indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(self):
        with torch.no_grad():
            while True:
                inputs = {'input_ids': self.generated}
                outputs = self.model(**inputs)
                next_token_logits = outputs[0][0, -1, :]
                filtered_logits = self.filtering(next_token_logits)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if next_token.tolist() == [4]:
                    break
                self.generated = torch.cat((self.generated, next_token.unsqueeze(0)), dim=1)
        return self.generated

model_path = '/data/disk2/private/hujinyi/poem_vae/model_no_tag_in_decoder/model_gpt2_keywords/model_epoch_10.pt'
model_config_path = '/data/disk2/private/hujinyi/poem_vae/data/config.json'
vocab_path = '/data/disk2/private/hujinyi/poem_vae/data/vocab.txt'

generator = Generator(model_path, model_config_path, vocab_path)
keywords = ['春天', '夏天', '秋天', '冬天']
genre = '七言绝句'
generator.generate(keywords, genre)
from transformers import BertForSequenceClassification, BertTokenizer, TextClassificationPipeline

class Rewarder:
    def __init__(self, model_dir, tokenizer_dir, device_id):
        model = BertForSequenceClassification.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device_id, return_all_scores=True)

    def reward(self, txt):
        lis = self.pipeline(txt)
        ret = []
        for ins in lis:
            for ele in ins:
                if ele['label'] == 'LABEL_1':
                    ret.append(ele['score'])
        assert len(ret) == len(txt)
        return ret

def main():
    poet_name = '杜甫'
    model_dir = f'/home/liwenhao/bert_new/models/Poets/{poet_name}'
    rew = Rewarder(model_dir, '/home/liwenhao/bert_new/models/v1', -1)
    print('load okay')
    txt = ['床前明月光，疑是地上霜，举头望明月，低头思故乡'] * 3 + ['国破山河在，城春草木深，感时花溅泪，恨别鸟惊心']
    print(rew.reward(txt))

if __name__ == '__main__':
    main()


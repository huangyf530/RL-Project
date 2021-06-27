import numpy as np
from tqdm import tqdm, trange
from Rewarder.Rewarder import Rewarder
import torch
import sys

# test_file = "data/test.txt"
# test_file = "output/gpt_v1.test.txt"
# test_file = "output/gpt_v1.valid.txt"
device = '/GPU:0' if torch.cuda.is_available() else '/cpu:0'


batch_size = 32
def read_poems(filename):
    poems = []
    with open(filename, 'r') as f:
        for linenum, line in enumerate(f):
            line = line.strip()
            if line == '':
                continue
            poem = line.split('|')[-1]
            # poem = poem.replace('，', '|')
            poems.append(poem)
    print(f"read {len(poems)} poem from {filename}")
    return poems

if __name__=="__main__":
    test_file = sys.argv[1]
    poems = read_poems(test_file)
    rewarder = Rewarder(sep_device=device)
    print(poems[:10])
    final_scores_list, lm_vals_list, mi_vals_list, tfidf_vals_list, quality_vals_list = [], [], [] , [], []
    for i in trange(0, len(poems), batch_size):
        end = min(i + batch_size, len(poems))
        input_poems = poems[i:end]
        final_scores, lm_vals, mi_vals, tfidf_vals, quality_vals = rewarder.get_mixed_scores_with_poems(input_poems, 4)
        final_scores_list.extend(final_scores.tolist())
        lm_vals_list.extend(lm_vals.tolist())
        mi_vals_list.extend(mi_vals.tolist())
        tfidf_vals_list.extend(tfidf_vals.tolist())
        quality_vals_list.extend(quality_vals.tolist())
    print(len(final_scores_list))
    print(len(lm_vals_list))
    print("final scores:", np.mean(final_scores_list))
    print("lm vals:", np.mean(lm_vals_list))
    print("mi vals:", np.mean(mi_vals_list))
    print("quality vals:", np.mean(quality_vals_list))
    print("tfidf vals:", np.mean(tfidf_vals_list))


import json
import os
import pickle
import torch


class GLConstants:

    def __init__(self, data_path):
        self.__get_sets(data_path + 'vocab_info.json')
        self.__get_patters()
        self.__load_rhyme_dic(data_path + "pingshui.txt", data_path + "pingshui_amb.pkl")

    def __get_sets(self, fn):
        ids = json.load(open(fn))
        print('unk_id', ids['unk_id'])
        self.sets = {'A': set(ids['zi_ids']),
                ',': {ids['comma_id']},
                'N': {ids['pad_id']},
                'P': set(ids['pingsheng_ids']),
                'Z': set(ids['zesheng_ids']),
                'U': {ids['unk_id']}}

    def __get_patters(self):
        self.__RHYTHM_TYPES = [[0, 1, 3, 2], [1, 2, 0, 1], [2, 1, 3, 2], [3, 2, 0, 1]]
        # self.__RHYTHM_TYPES = [[0, 1], [3, 2]]
        self.__RHYTHM_PATTERNS = {7: ['AZAPPZZ', 'APAZZPY', 'AZPPZZY', 'APAZPPZ'],
                                  5: ['APPZZ','AZZPY', 'PPZZY', 'AZPPZ']}
    def __load_rhyme_dic(self, rhyme_dic_path, rhyme_disamb_path):

        self.__rhyme_dic = {} # char id to rhyme category ids
        self.__rhyme_idic = {} # rhyme category id to char ids

        with open(rhyme_dic_path, 'r') as fin:
            lines = fin.readlines()

        amb_count = 0
        for line in lines:
            (char, rhyme_id) = line.strip().split(' ')
            char_id = char
            rhyme_id = int(rhyme_id)

            if not char_id in self.__rhyme_dic:
                self.__rhyme_dic.update({char_id:[rhyme_id]})
            elif not rhyme_id in self.__rhyme_dic[char_id]:
                self.__rhyme_dic[char_id].append(rhyme_id)
                amb_count += 1

            if not rhyme_id in self.__rhyme_idic:
                self.__rhyme_idic.update({rhyme_id:[char_id]})
            else:
                self.__rhyme_idic[rhyme_id].append(char_id)

        print ("  rhyme dic loaded, ambiguous rhyme chars: %d" % (amb_count))

        # load data for rhyme disambiguation
        self.__ngram_rhyme_map = {} # rhyme id list of each bigram or trigram
        self.__char_rhyme_map = {} # the most likely rhyme id for each char
        # load the calculated data, if there is any
        #print (rhyme_disamb_path)
        assert rhyme_disamb_path is not None and os.path.exists(rhyme_disamb_path)

        with open(rhyme_disamb_path, 'rb') as fin:
            self.__char_rhyme_map = pickle.load(fin)
            self.__ngram_rhyme_map = pickle.load(fin)

            print ("  rhyme disamb data loaded, cached chars: %d, ngrams: %d"
                % (len(self.__char_rhyme_map), len(self.__ngram_rhyme_map)))

    def get_line_rhyme(self, line):
        """ we use statistics of ngram to disambiguate the rhyme category,
        but there is still risk of mismatching and ambiguity
        """
        tail_char = line[-1]

        if tail_char in self.__char_rhyme_map:
            bigram = line[-2] + line[-1]
            if bigram in self.__ngram_rhyme_map and len(self.__ngram_rhyme_map[bigram]) == 1:
                return self.__ngram_rhyme_map[bigram][0]

            trigram = line[-3] + line[-2] + line[-1]
            if trigram in self.__ngram_rhyme_map and len(self.__ngram_rhyme_map[bigram]) == 1:
                return self.__ngram_rhyme_map[trigram][0]

            return self.__char_rhyme_map[tail_char][0]

        if tail_char in self.__rhyme_dic:
            return self.__rhyme_dic[tail_char][0]

        return -1

    def get_rhythm_chars(self, idx):
        return self.__rhyme_idic[idx]

    def get_seq(self, yan, idx):
        typ = self.__RHYTHM_TYPES[idx]
        return ','.join([self.__RHYTHM_PATTERNS[yan][t] for t in typ])


class GLController:

    constants = GLConstants('./data/')

    def __init__(self, yan, pattern_id, tokenizer):
        self.yan = yan
        self.tokenizer = tokenizer
        self.i = -1
        self.seq = self.constants.get_seq(yan, pattern_id)
        self.rep_ids = []
        self.yun_id = None

    def step(self):
        self.i += 1
        if self.i != 0 and self.i < len(self.seq) and self.seq[self.i - 1] == 'Y' and (self.yun_id is None or self.yun_id == -1):
            self.yun_id = int(self.get_yun_id())
            # print(self.tokenizer.decode(self.rep_ids), self.yun_id)
        if self.i >= len(self.seq):
            ok_ids = self.constants.sets['N']
        elif self.seq[self.i] == 'Y':
            if self.yun_id is None or self.yun_id == -1:
                ok_ids = self.constants.sets['P']
            else:
                ok_ids = self.get_yun()
        else:
            ok_ids = self.constants.sets[self.seq[self.i]]
        ok_ids = ok_ids.difference(self.constants.sets['U'])
        assert len(ok_ids) >= 1
        if len(ok_ids) > 1:
            ok_ids = ok_ids.difference(set(self.rep_ids))
        return list(ok_ids)

    def get_yun_id(self):
        first_sen = self.tokenizer.decode(self.rep_ids)
        yun_id = self.constants.get_line_rhyme(first_sen)
        return yun_id

    def get_yun(self):
        yun_chars = self.constants.get_rhythm_chars(self.yun_id)
        next_sen = self.tokenizer.decode(self.rep_ids)
        ret = []
        for c in yun_chars:
            next_sen_tmp = next_sen + c
            if self.constants.get_line_rhyme(next_sen_tmp) == self.yun_id:
                ret.append(self.tokenizer.convert_tokens_to_ids(list(c))[0])
        if len(ret) != 0:
            return set(ret)
        else:
            return self.constants.sets['P']



    def update_token(self, token):
        self.rep_ids.append(token)


class BatchMasker:

    def __init__(self, yans, idxs, tokenizer, vocab_size, device):
        self.vocab_size = vocab_size
        self.device = device
        self.controllers = [GLController(yans[i], idxs[i], tokenizer) for i in range(len(yans))]

    def lis2occ(self, lis):
        ret = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        ret[lis] = True
        return ret

    def step(self):
        ret = []
        for i in range(len(self.controllers)):
            s = self.controllers[i].step()
            mask = self.lis2occ(s)
            ret.append(mask.unsqueeze(0))
        return torch.cat(ret, dim=0)

    def update_tokens(self, tokens):
        tokens = tokens.detach().cpu().numpy().reshape(-1)
        for i in range(len(self.controllers)):
            self.controllers[i].update_token(tokens[i])




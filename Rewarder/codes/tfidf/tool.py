import pickle
import numpy as np
import random
import copy

class Tool(object):
    def __init__(self, sen_len):
        self.__sen_len = sen_len
    
    def line2words(self, line):
        line = line.strip()
        words= [c for c in line]
        return words

    def words2idxes(self, words):
        ''' Characters to idx list '''
        idxes = []
        for w in words:
            if w in self.__vocab:
                idxes.append(self.__vocab[w])
            else:
                idxes.append(self.__vocab['UNK'])
        return idxes

    def idxes2words(self, idxes, omit_special=True):
        words = []
        for idx in idxes:
            if  (idx == self.__PAD_ID or idx == self.__B_ID 
                or idx == self.__E_ID or idx == self.__M_ID) and omit_special:
                continue
            words.append(self.__ivocab[idx])

        return words

    def get_vocab(self):
        return copy.deepcopy(self.__vocab)

    def get_ivocab(self):
        return copy.deepcopy(self.__ivocab)

    def get_vocab_size(self):
        if self.__vocab:
            return len(self.__vocab)
        else:
            return -1

    def get_PAD_ID(self):
        assert self.__vocab is not None
        return self.__PAD_ID

    def get_special_IDs(self):
        assert self.__vocab is not None
        return self.__PAD_ID, self.__UNK_ID, self.__B_ID, self.__E_ID

    def greedy_search(self, outputs):
        outidx = [int(np.argmax(logit, axis=0)) for logit in outputs]
        #print (outidx)
        if self.__E_ID in outidx:
            outidx = outidx[:outidx.index(self.__E_ID)]

        words = self.idxes2words(outidx)
        sentence = " ".join(words)
        return sentence

    def load_dic(self, vocab_path, ivocab_path):
        vocab_file = open(vocab_path, 'rb')
        dic = pickle.load(vocab_file)
        vocab_file.close()

        ivocab_file = open(ivocab_path, 'rb')
        idic = pickle.load(ivocab_file)
        ivocab_file.close()

        self.__vocab = dic
        self.__ivocab = idic

        self.__PAD_ID = dic['PAD']
        self.__UNK_ID = dic['UNK']
        self.__E_ID = dic['<E>']
        self.__B_ID = dic['<B>']
        self.__M_ID = dic['<M>']

    def build_data(self, data_path, batch_size):
        '''
        Build data as batches.
        NOTE: Please run load_dic() at first.
        '''
        corpus_file = open(data_path, 'rb')
        corpus = pickle.load(corpus_file)
        corpus_file.close()

        # TMP
        N = int(0.005*len(corpus))
        corpus = corpus[0:N]
        batches, batch_num = self.build_batches(corpus, batch_size) 

        return batches, batch_num

    def build_batches(self, data, batch_size):
        batched_data = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))  
        for bi in range(0, batch_num):
            instances = data[bi*batch_size : (bi+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size - len(instances))

            # Build all batched data
            sens = [instance[0] for instance in instances] 
            vals = [instance[1] for instance in instances] 

            enc_inps, enc_len_inps = self.get_batch_enc_sen(sens, batch_size)

            data_dic = {}
            data_dic['enc_inps'] = enc_inps
            data_dic['enc_len_inps'] = enc_len_inps
            data_dic['labels'] = vals 

            batched_data.append(data_dic)

        return batched_data, batch_num

    def get_batch_enc_sen(self, outputs, batch_size):
        assert  len(outputs) == batch_size
        enc_inps, len_inps = [], []

        for i in range(batch_size):
            enc_inp = outputs[i]
            len_inps.append(len(enc_inp))
            enc_pad_size = self.__sen_len - len(enc_inp)
            enc_pad = [self.__PAD_ID] * enc_pad_size
            enc_inps.append(enc_inp + enc_pad)

        # Now we create batch-major vectors from the data selected above.
        batch_enc_inps = []

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(self.__sen_len):
            batch_enc_inps.append(np.array([enc_inps[batch_idx][length_idx]
                                                  for batch_idx in range(batch_size)], dtype=np.int32))

        return batch_enc_inps, len_inps

    # -----------------------------------------------------------
    # Tools for beam search
    def get_eval_batch_enc_sen(self, ori_sens):
        data = []
        for sen in ori_sens:
            sen = sen.strip()
            sen = sen.replace("UNK", "U")
            sen = sen.replace(" ", "")
            chars = self.line2words(sen)
            idxes = self.words2idxes(chars)
            data.append(idxes)
    
        batch_size = len(data)
        enc_inps, len_inps = [], []

        for i in range(batch_size):
            enc_inp = data[i]
            len_inps.append(len(enc_inp))
            enc_pad_size = self.__sen_len - len(enc_inp)
            enc_pad = [self.__PAD_ID] * enc_pad_size
            enc_inps.append(enc_inp + enc_pad)

        # Now we create batch-major vectors from the data selected above.
        batch_enc_inps = []

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(self.__sen_len):
            batch_enc_inps.append(np.array([enc_inps[batch_idx][length_idx]
                                                  for batch_idx in range(batch_size)], dtype=np.int32))

        return batch_enc_inps, len_inps
    
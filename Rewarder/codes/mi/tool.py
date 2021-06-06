import pickle
import numpy as np
import random
import copy

class Tool(object):
    def __init__(self, enc_len, dec_len):
        self.__enc_len = enc_len
        self.__dec_len = dec_len
    
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
            enc_sens = [instance[0] for instance in instances] 
            dec_sens = [instance[1] for instance in instances] 

            enc_inps, enc_len_inps = self.get_batch_enc_sen(enc_sens, batch_size)

            dec_inps, dec_weights = self.get_batch_dec_sen(dec_sens, batch_size)

            data_dic = {}
            data_dic['enc_inps'] = enc_inps
            data_dic['enc_len_inps'] = enc_len_inps

            data_dic['dec_inps'] = dec_inps
            data_dic['trg_weights'] = dec_weights
            batched_data.append(data_dic)

        return batched_data, batch_num

    def get_batch_enc_sen(self, outputs, batch_size):
        assert  len(outputs) == batch_size
        enc_inps, len_inps = [], []

        for i in range(batch_size):
            enc_inp = outputs[i]
            len_inps.append(len(enc_inp))
            enc_pad_size = self.__enc_len - len(enc_inp)
            enc_pad = [self.__PAD_ID] * enc_pad_size
            enc_inps.append(enc_inp + enc_pad)

        # Now we create batch-major vectors from the data selected above.
        batch_enc_inps = []

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(self.__enc_len):
            batch_enc_inps.append(np.array([enc_inps[batch_idx][length_idx]
                                                  for batch_idx in range(batch_size)], dtype=np.int32))

        return batch_enc_inps, len_inps

    def get_batch_dec_sen(self, outputs, batch_size):
        assert len(outputs) == batch_size
        dec_inps = []

        for i in range(batch_size):
            dec_inp = outputs[i] + [self.__E_ID]

            # Decoder inputs get an extra "<B>" symbol, and are padded then.
            dec_pad_size = self.__dec_len - len(dec_inp) - 1
            dec_inps.append([self.__B_ID] + dec_inp + [self.__PAD_ID] * dec_pad_size)

        # Create batch-major vectors.
        batch_dec_inps, batch_weights = [], []

        for length_idx in range(self.__dec_len):
            batch_dec_inps.append(np.array([dec_inps[batch_idx][length_idx]
                                                  for batch_idx in range(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float32)
            for batch_idx in range(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < self.__dec_len - 1:
                    target = dec_inps[batch_idx][length_idx + 1]
                if length_idx == self.__dec_len - 1 or target == self.__PAD_ID:
                    batch_weight[batch_idx] = 0.0

            batch_weights.append(batch_weight)

        return batch_dec_inps, batch_weights

    # -----------------------------------------------------------
    # Tools for beam search
    def beam_get_sentence(self, idxes):
        if idxes is not list:
            idxes = list(idxes)
        if self.__E_ID in idxes:
          idxes = idxes[:idxes.index(self.__E_ID)]

        chars = self.idxes2chars(idxes)
        sentence = "".join(chars)

        return sentence

    def gen_eval_enc(self, sentence, batch_size):
        enc_inps, len_inps = [], []

        for i in range(0, batch_size):
            enc_inp = sentence
            enc_pad_size = self.__sen_len - len(enc_inp)
            enc_pad = [self.__PAD_ID] * enc_pad_size
            enc_inps.append(enc_inp + enc_pad)
            len_inps.append(len(enc_inp))


        batch_enc_inps= []
        for length_idx in range(self.__sen_len):
            batch_enc_inps.append(
                np.array([enc_inps[batch_idx][length_idx]
                    for batch_idx in range(batch_size)], dtype=np.int32))
        
        return batch_enc_inps, len_inps
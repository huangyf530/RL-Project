import pickle
import numpy as np
import random
import copy
import math


class DataTool(object):
    """docstring for Tool"""
    def __init__(self,):
        pass
        

    '''
    split line to character and save as list
    '''
    def line2chars(self, line):
        line = line.strip()
        chars = [c for c in line]
        return chars

    ''' char list to idx list '''
    def chars2idxes(self, chars):
        idxes = []
        for c in chars:
            if c in self.vocab:
                idxes.append(self.vocab[c])
            else:
                idxes.append(self.vocab['UNK'])
        return idxes

    def get_vocab(self):
        return copy.deepcopy(self.vocab)
    def get_ivocab(self):
        return copy.deepcopy(self.ivocab)

    def get_vocab_size(self):
        if self.vocab:
            return len(self.vocab)
        else:
            return -1

    def get_PAD_ID(self):
        if self.vocab:
            return self.PAD_ID
        else:
            return -1

    def get_E_ID(self):
        if self.vocab:
            return self.E_ID
        else:
            return -1

    def get_B_ID(self):
        if self.vocab:
            return self.B_ID
        else:
            return -1

    def load_dic(self, file_dir):
        vocab_file = open(file_dir + '/vocab.pickle', 'rb')
        dic = pickle.load(vocab_file)
        vocab_file.close()

        ivocab_file = open(file_dir + '/ivocab.pickle', 'rb')
        idic = pickle.load(ivocab_file)
        ivocab_file.close()

        self.vocab = dic
        self.ivocab = idic

        self.E_ID = dic['<E>']
        self.PAD_ID = dic['PAD']
        self.B_ID = dic['<B>']
        self.UNK_ID = dic['UNK']
        self.M_ID = dic['<M>']

    def build_data(self, file_path, batch_size):
        corpus_file = open(file_path, 'rb')
        data = pickle.load(corpus_file)
        corpus_file.close()

        #TMP
        print (len(data))
        data = data[0:500]

        batched_data, batch_num = self.build_batches(data,batch_size)
        return batched_data, batch_num, len(data)

    def build_batches(self, raw_data, batch_size):
        batch_num = int(math.ceil(len(raw_data) / float(batch_size)))
        data = []
        for i in range(0, batch_num):
            sens = raw_data[i*batch_size : (i+1)*batch_size]
            #print (np.shape(sens))
            #print (sens[0])
            max_len = max([len(sen) for sen in sens])
            batch_x = []
            batch_y = []
            for sen in sens:
                #print (sen)
                pad_size = max_len - len(sen)
                new_sen = sen + [self.PAD_ID]*pad_size
                #print (new_sen)
                x = new_sen[0:len(new_sen)-1]
                y = new_sen[1:]
                assert len(x) == len(y) == max_len-1
                
                batch_x.append(x)
                batch_y.append(y)
                #print (x)
                #print (y)
                #input(">")

            data.append((batch_x, batch_y))

        return data, batch_num

    def build_one_batch(self, sens):
        max_len = max([len(sen) for sen in sens])
        batch_x = []
        batch_y = []
        for sen in sens:
            #print (sen)
            pad_size = max_len - len(sen)
            new_sen = sen + [self.PAD_ID]*pad_size
            #print (new_sen)
            x = new_sen[0:len(new_sen)-1]
            y = new_sen[1:]
            assert len(x) == len(y) == max_len-1
                
            batch_x.append(x)
            batch_y.append(y)
        
        return batch_x, batch_y



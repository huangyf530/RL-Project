import tensorflow as tf
import csv
import pickle
import numpy as np
import random
import copy

'''
Returns vocab frequencies.
Returns: List of integers, length=FLAGS.vocab_size.
Raises: ValueError: if the length of the frequency file is not equal to the vocab
size, or if the file is not found.
'''
def get_vocab_freqs(path, vocab_size):
    if tf.gfile.Exists(path):
        with tf.gfile.Open(path) as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            freqs = [int(row[-1]) for row in reader]
            if len(freqs) != vocab_size:
                raise ValueError('Frequency file length %d != vocab size %d' %
                             (len(freqs), vocab_size))
    else:
        raise ValueError('vocab_freq_path not found')

    return freqs

'''
loading  training data, including vocab, inverting vocab and corpus
'''
def load_data(file_dir, mode, only_vocab=False):
    if mode == 'pre_train':
        prefix = 'lm'
    else:
        prefix = 'cl'

    vocab_file = open(file_dir + '/vocab.pkl', 'rb')
    dic = pickle.load(vocab_file)
    vocab_file.close()

    ivocab_file = open(file_dir + '/ivocab.pkl', 'rb')
    idic = pickle.load(ivocab_file)
    ivocab_file.close()

    if only_vocab:
        return dic, idic

    corpus_file = open(file_dir + '/'  + 'train_' + prefix + '.pickle', 'rb')
    corpus = pickle.load(corpus_file)
    corpus_file.close()

    corpus_file = open(file_dir + '/'  + 'valid_' + prefix + '.pickle', 'rb')
    valid_corpus = pickle.load(corpus_file)
    corpus_file.close()

    return dic, idic, corpus, valid_corpus


def get_batch_sentence_lm(inputs, batch_size, EOS_ID, PAD_ID, reverse=False):
    seqinps, weights, lm_labels  = [], [], []
    assert len(inputs) == batch_size
    lengths = [len(inpseq)+1 for inpseq in inputs] 
    #max_len = max(lengths)
    max_len = 30
    for i in range(batch_size):
        seq = copy.deepcopy(inputs[i])
        if reverse:
            seq.reverse()

        l = lengths[i]
        inp =  seq + [EOS_ID] + [PAD_ID] * (max_len-l)
        weights.append([1.0]*(l-1) + [0.0]*(max_len-l+1))
        seqinps.append(inp)

        lm_labels.append(seq[1:] + [EOS_ID] + [PAD_ID] * (max_len-l+1))


    return seqinps, weights, lengths, lm_labels

def get_batch_sentence_cl(inputs, ori_labels, batch_size, EOS_ID, PAD_ID):
    seqinps, iseqinps, weights, cl_labels  = [], [], [], []
    assert len(inputs) == batch_size
    lengths = [len(inpseq)+1 for inpseq in inputs] 
    #max_len = max(lengths)
    max_len = 30
    for i in range(batch_size):
        seq = copy.deepcopy(inputs[i])
        iseq = copy.deepcopy(inputs[i])
        iseq.reverse()

        l = lengths[i]
        seqinps.append(seq + [EOS_ID] + [PAD_ID] * (max_len-l))
        iseqinps.append(iseq + [EOS_ID] + [PAD_ID] * (max_len-l))

        weight = [0.0] * max_len
        weight[l-1] = 1.0
        weights.append(weight)

        cl_label = [0] * max_len
        cl_label[l-1] = ori_labels[i] 
       
        cl_labels.append(cl_label)
        #tt = input(">")

    return seqinps, iseqinps, weights, lengths, cl_labels

def build_batches(data, batch_size, EOS_ID, PAD_ID, mode):
        batched_data = []
        batch_num = int(np.ceil(len(data) / batch_size))

        assert mode == 'pre_train' or mode == 'fine_tune'

        for i in range(0, batch_num):
            instances = data[i*batch_size : (i+1)*batch_size]
            if len(instances) < batch_size:
                instances = instances + random.sample(data, batch_size-len(instances))

            inputs = [instance[0] for instance in instances]

            #print (inputs)
            #input(">")

            if mode == 'fine_tune':
                labels = [instance[1] for instance in instances]
                seqinps, iseqinps, weights, lengths, cl_labels = \
                    get_batch_sentence_cl(inputs, labels, batch_size, EOS_ID, PAD_ID)
                batched_data.append((seqinps, iseqinps, weights, lengths, cl_labels, labels))
            else:
                seqinps, weights, lengths, lm_labels = \
                    get_batch_sentence_lm(inputs, batch_size, EOS_ID, PAD_ID)
                iseqinps, iweights, ilengths, ilm_labels = \
                    get_batch_sentence_lm(inputs, batch_size, EOS_ID, PAD_ID, True)
                
                batched_data.append(((seqinps, weights, lengths, lm_labels), 
                    (iseqinps, iweights, ilengths, ilm_labels)))

        return batched_data, batch_num

def build_data(file_dir, batch_size, mode='pre_train'):
    vocab, ivocab, train_data, valid_data = load_data(file_dir, mode)

    #TMP
    train_data = train_data[0:500]
    valid_data = valid_data[0:100]

    EOS_ID = vocab['</S>']
    PAD_ID = vocab['PAD']
    train_batches, train_batch_num = build_batches(
        train_data, batch_size, EOS_ID, PAD_ID, mode) 
    valid_batches, valid_batch_num = build_batches(
        valid_data, batch_size, EOS_ID, PAD_ID, mode)

    return vocab, ivocab, train_batch_num, \
        valid_batch_num, train_batches, valid_batches

def build_predict_data(file_dir, test_file, batch_size):
    vocab, ivocab = load_data(file_dir, 'fine_tune', only_vocab=True)

    EOS_ID = vocab['</S>']
    PAD_ID = vocab['PAD']

    # build seq
    fin = open(test_file, 'r')
    lines = fin.readlines()
    fin.close()

    random.shuffle(lines)
    N = 10000
    lines = lines[0:N]

    data = []
    skip_count = 0
    for line in lines:
        line = line.strip()
        if line.find("f") != -1 or line.find("er") != -1:
            skip_count += 1
            continue
        else:
            #print (line)
            poem = line.replace("|", "")
            poem = poem.replace("UNK", "u")
            chars = [c for c in poem]
        if len(chars) > 28:
            print (len(chars))
            print (line)
            skip_count += 1
            continue
        idxes = [vocab[c] if c in vocab else vocab['UNK'] for c in chars]
        data.append(idxes)

    #
    batched_data = []
    batch_num = int(np.ceil(len(data) / batch_size))
    ori_data_num = len(data)

    for i in range(0, batch_num):
        instances = data[i*batch_size : (i+1)*batch_size]
        if len(instances) < batch_size:
            instances = instances + random.sample(data, batch_size-len(instances))

        inputs = instances
        labels = [0 for instance in instances]
        seqinps, iseqinps, weights, lengths, cl_labels = \
            get_batch_sentence_cl(inputs, labels, batch_size, EOS_ID, PAD_ID)
        batched_data.append((seqinps, iseqinps, weights, lengths, cl_labels))

    return vocab, batched_data, batch_num, ori_data_num

def build_test_data(file_dir, test_file, batch_size):
    vocab, ivocab = load_data(file_dir, 'fine_tune', only_vocab=True)

    EOS_ID = vocab['</S>']
    PAD_ID = vocab['PAD']

    # build seq
    fin = open(test_file, 'r')
    lines = fin.readlines()
    fin.close()

    data = []
    skip_count = 0
    for line in lines:
        line = line.strip()
        if line.find("f") != -1 or line.find("er") != -1:
            skip_count += 1
            continue
        else:
            #print (line)
            para = line.split("#")
            poem = para[0].replace("|", "")
            poem = poem.replace("UNK", "u")
            chars = [c for c in poem]
            label = int(para[1])
        if len(chars) > 28:
            print (len(chars))
            print (line)
        idxes = [vocab[c] if c in vocab else vocab['UNK'] for c in chars]
        data.append((idxes, label))

    #
    batched_data = []
    batch_num = int(np.ceil(len(data) / batch_size))
    ori_data_num = len(data)

    for i in range(0, batch_num):
        instances = data[i*batch_size : (i+1)*batch_size]
        if len(instances) < batch_size:
            instances = instances + random.sample(data, batch_size-len(instances))

        inputs = [instance[0] for instance in instances]
        labels = [instance[1] for instance in instances]
        seqinps, iseqinps, weights, lengths, _ = \
            get_batch_sentence_cl(inputs, labels, batch_size, EOS_ID, PAD_ID)
        batched_data.append((seqinps, iseqinps, weights, lengths, labels))

    return vocab, batched_data, batch_num, ori_data_num


#--------------------------------------------------------------------------------------------------
'''
poems: list of poem, each poem is seperated by '|', withoud label
'''
def build_predict_batch_by_chars(poems, vocab):

    EOS_ID = vocab['</S>']
    PAD_ID = vocab['PAD']

    inputs = []
    for chars in poems:
        idxes = [vocab[c] if c in vocab else vocab['UNK'] for c in chars]
        inputs.append(idxes)

    lengths = [len(inpseq)+1 for inpseq in inputs] 
    max_len = 30
    seqinps, iseqinps = [], []
    for i in range(len(inputs)):
        seq = copy.deepcopy(inputs[i])
        iseq = copy.deepcopy(inputs[i])
        iseq.reverse()

        l = lengths[i]
        seqinp = seq + [EOS_ID] + [PAD_ID] * (max_len-l)
        iseqinp = iseq + [EOS_ID] + [PAD_ID] * (max_len-l)

        seqinps.append(seqinp)
        iseqinps.append(iseqinp)

    return seqinps, iseqinps, lengths

def build_test_batch(poems, vocab, ivocab):

    EOS_ID = vocab['</S>']
    PAD_ID = vocab['PAD']

    inputs = []
    for line in poems:
        if line.find('f') != -1:
            continue
        line = line.strip().replace(" ", "")
        line = line.replace("|", "")
        chars = [c for c in line]
        idxes = [vocab[c] if c in vocab else vocab['UNK'] for c in chars]

        inputs.append(idxes)

    lengths = [len(inpseq)+1 for inpseq in inputs] 
    max_len = 30
    seqinps, iseqinps = [], []
    for i in range(len(inputs)):
        seq = copy.deepcopy(inputs[i])
        iseq = copy.deepcopy(inputs[i])
        iseq.reverse()

        l = lengths[i]
        seqinp = seq + [EOS_ID] + [PAD_ID] * (max_len-l)
        iseqinp = iseq + [EOS_ID] + [PAD_ID] * (max_len-l)

        seqinps.append(seqinp)
        iseqinps.append(iseqinp)

    return seqinps, iseqinps, lengths

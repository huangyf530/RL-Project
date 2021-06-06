from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pickle
import os
import numpy as np

import tensorflow as tf
import Rewarder.codes.dis.graphs as graphs
import Rewarder.codes.dis.tool as tool

''''construction for ClassifierTraine'''
class ClassifierEvaluator(object):
    
    def __init__(self, graph, sep_device):

        self.root_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
        self.device = sep_device
        self.tool = tool

        # load dic
        self.vocab, self.ivocab = self.load_dic(self.root_dir + "data/dis")

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
        gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(graph = graph, 
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        
        self.model = graphs.VatxtModel(self.root_dir + "data/dis/freq_vocab.txt", 
            len(self.vocab), self.vocab['</S>'], self.device)
        _, self.probs_op = self.model.eval_graph()
        self.load_model(self.sess, self.root_dir+"model/dis/")

    def load_dic(self, file_dir):

        vocab_file = open(file_dir + '/vocab.pkl', 'rb')
        dic = pickle.load(vocab_file)
        vocab_file.close()

        ivocab_file = open(file_dir + '/ivocab.pkl', 'rb')
        idic = pickle.load(ivocab_file)
        ivocab_file.close()

        return dic, idic

    def load_model(self, session, model_dir):
        """load model."""
        saver = tf.train.Saver(tf.global_variables() , write_version=tf.train.SaverDef.V1)
        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                  ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise ValueError("%s not found! " % ckpt.model_checkpoint_path)


    def score_batch(self, poems_in_chars):
        '''
        Input: poems, list, each element a poem with chars
        '''
        #batch_size = len(poems_in_chars)
        batch = self.tool.build_predict_batch_by_chars(poems_in_chars, self.vocab)
        seq, iseq, length = batch

        input_feed = {}
        input_feed[self.model.cl_inputs_f['x'].name] = seq
        input_feed[self.model.cl_inputs_r['x'].name] = iseq
        input_feed[self.model.cl_inputs_f['length'].name] = length
        
        [probs_val]  = self.sess.run([self.probs_op], input_feed)

        #print (np.shape(probs_val))
        #print (probs_val)
        level = np.array([1.0, 2.0, 3.0])
        level = np.reshape(level, (3,1))
        scores = np.matmul(probs_val, level)
        scores = list(scores[:, 0])

        #print (scores)
        #input(">")

        return scores


    def predict(self, file, required_label):
        batch_size = 32

        vocab, test_data, _, ori_data_num = tool.build_test_data("data/", file, batch_size)
        print ("test_data_num: %d" % (len(test_data)))

        model = graphs.VatxtModel("data/freq_vocab.txt", len(vocab), vocab['</S>'], self.device)
        pred,_ = model.eval_graph(batch_size=batch_size)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
        gpu_options.allow_growth = True

        pred_vec = []

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            self.load_model(sess, "model/")

            for i, batch in enumerate(test_data):
                if i % 50 == 0:
                    print ("%d/%d" % (i, len(test_data)))

                seq, iseq, _, length, _ = batch
                input_feed = {}
                input_feed[model.cl_inputs_f['x'].name] = seq
                input_feed[model.cl_inputs_r['x'].name] = iseq
                input_feed[model.cl_inputs_f['length'].name] = length
                [pred_val]  = sess.run([pred], input_feed)
                pred_vec.extend(pred_val)

            print (len(pred_vec))
            pred_vec = np.array(pred_vec[0:ori_data_num])
            print (len(pred_vec))
            print (pred_vec)
            ratio = pred_vec == required_label
            ratio = np.sum(ratio.astype(np.float32)) / ori_data_num
            print ("predict ratio:%.4f" % (ratio))


def main(_):
    trainer = ClassifierEvaluator()
    #trainer.test_file("cl_valid.txt")
    trainer.score_file("testdata/test_gen_greedy.txt")
    #trainer.predict2("basic_times_1.txt", 1)
    #trainer.predict("save/adv_outs_fine_11e_21_.txt", 1)
 

if __name__ == "__main__":
    tf.app.run()

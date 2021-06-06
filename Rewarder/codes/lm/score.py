from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from Rewarder.codes.lm.graph import LMModel
from Rewarder.codes.lm.tool import DataTool
from Rewarder.codes.lm.config import hps

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class LMScorer(object):
    """construction for PoemTrainer"""
    def __init__(self, graph, sep_device):
        self.__root_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

        self.hps = hps
        self.data_tool = DataTool()
        self.data_tool.load_dic(self.__root_dir + "../../data/lm/")
        vocab_size = self.data_tool.get_vocab_size()
        assert vocab_size > 0
        self.hps = self.hps._replace(vocab_size=vocab_size)
        self.hps = self.hps._replace(device=sep_device)

        self.ivocab = self.data_tool.get_ivocab()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
        gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(graph=graph, 
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        
        self.model = LMModel(self.hps)
        self.model.build_eval_graph()
        self.load_model(self.__root_dir + "../../model/lm/")

    def load_model(self, model_dir):
        """load model."""
        saver = tf.train.Saver(tf.global_variables() , write_version=tf.train.SaverDef.V1)
        
        #print (model_dir)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                    ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("%s not found! " % ckpt.model_checkpoint_path)


    def preprocess_sen(self, sens):
        B_ID = self.data_tool.get_B_ID()
        E_ID = self.data_tool.get_E_ID()
        idxes_vec = []
        for sen in sens:
            chars = self.data_tool.line2chars(sen.strip())
            idxes = self.data_tool.chars2idxes(chars)
            idxes = [B_ID] + idxes + [E_ID]
            idxes_vec.append(idxes)

        batch_x, batch_y = self.data_tool.build_one_batch(idxes_vec)
        return batch_x, batch_y


    def get_ppl(self, sens, return_ppl=False):
        x, y = self.preprocess_sen(sens)
        input_feed = {}
        input_feed[self.model.keep_prob] = 1.0
        input_feed[self.model.inputs.name] = x
        input_feed[self.model.targets.name] = y

        
        output_feed = [self.model.probs, self.model.targets_mask]
        outputs = self.sess.run(output_feed, input_feed)
        probs = outputs[0]
        mask = outputs[1]
        #print (np.shape(probs))
        #print (np.shape(mask))
        #print (np.shape(y))
        

        batch_size = len(sens)
        max_len = np.shape(probs)[1]
        cost = np.zeros((batch_size,1))
        y = np.array(y)

        #print (np.shape(probs))
        #print (np.shape(mask))
        #print (np.shape(y))

        #input(">")

        for i in range(0, max_len):
            #print ("%d %f" % (i, logits[i, idx]))
            #tt *= logits[i, idx]

            trgs = y[:, i]

            log_probs = np.log(probs[:, i, :]+1e-12)

            nll = np.diag(log_probs[:, trgs]) 

            current_nll = np.expand_dims(nll, axis=1)
            m = np.expand_dims(mask[:, i], axis=1)

            cost += np.multiply(current_nll, m)


        total_size = np.sum(mask, axis=1, keepdims=True)
        cost = cost / (total_size+1e-12)

        cost = cost[:, 0]

        if return_ppl:
            ppl = np.exp(-cost)
            return ppl
        else:
            return cost

    def get_next_prob(self, sen, num):
        x, y = self.preprocess_sen([sen])
        input_feed = {}
        input_feed[self.model.keep_prob] = 1.0
        input_feed[self.model.inputs.name] = x
        
        output_feed = [self.model.probs]
        outputs = self.sess.run(output_feed, input_feed)
        outputs = outputs[0]
        #print (np.shape(outputs))
        probs = outputs[0][-1]
        idxes = np.argsort(probs)
        idxes = list(idxes)
        idxes.reverse()

        idxes = idxes[0:num]

        vec = []
        for idx in idxes:
            vec.append((self.ivocab[idx], probs[idx]))

        return vec



if __name__ == "__main__":
    tf.app.run()

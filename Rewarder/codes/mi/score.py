from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import tensorflow as tf

import Rewarder.codes.mi.graphs as graphs
from Rewarder.codes.mi.tool import Tool
from Rewarder.codes.mi.config import hps


''''construction for Evaluator'''
class MIEvaluator(object):
    
    def __init__(self, graph, sep_device):
        self.__root_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
        self.hps = hps

        # construct HParams
        self.tool = Tool(enc_len=hps.enc_len, dec_len=hps.dec_len)
        self.tool.load_dic(self.__root_dir + hps.vocab_path, self.__root_dir + hps.ivocab_path)

        vocab_size = self.tool.get_vocab_size()
        assert vocab_size > 0
        self.hps = self.hps._replace(vocab_size=vocab_size)
        self.hps = self.hps._replace(device=sep_device)

        self.enc_len = hps.enc_len
        self.dec_len = hps.dec_len

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
        gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(graph=graph, 
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        self.model = graphs.Seq2Seq(self.hps)
        self.model.build_eval_graph()
        self.load_model()

    def load_model(self):
        """load model."""
        saver = tf.train.Saver(tf.global_variables() , write_version=tf.train.SaverDef.V1)

        ckpt = tf.train.get_checkpoint_state(self.__root_dir+"model/mi/")
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                    ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("%s not found! " % ckpt.model_checkpoint_path)


    def score(self, srcs, trgs):
        '''
        srcs: contains '|' symbols
        '''
        assert len(srcs) == len(trgs)
        src_idxes, trg_idxes = [], []
        for (src, trg) in zip(srcs, trgs):
            src_chars = []
            for c in src:
                if c == '|':
                    src_chars.append('<M>')
                else:
                    src_chars.append(c)
            trg_chars = [c for c in trg]
            src_idxes.append(self.tool.words2idxes(src_chars))
            trg_idxes.append(self.tool.words2idxes(trg_chars))


        enc_inps, len_inps = self.tool.get_batch_enc_sen(src_idxes, len(src_idxes))
        dec_inps, weights = self.tool.get_batch_dec_sen(trg_idxes, len(trg_idxes))

        batch_size = len(srcs)

        init_dec_state, memory = self.init_dec_state_computer(enc_inps, len_inps, batch_size)
        
        costs = np.zeros(batch_size, dtype=np.float32)
        state = init_dec_state

        for k in range(0, self.dec_len-1):
            inp = dec_inps[k]
            output, state = self.dec_state_computer(inp, state, memory, 
                enc_inps, len_inps, batch_size)
            
            log_probs = np.log(output)
            p = np.diag(log_probs[:, dec_inps[k+1]])
            costs += p * weights[k]

        total_size = np.sum(weights, axis=0) + 1e-12

        fin_cost = costs / total_size
        return fin_cost


# ------------------------
    def init_dec_state_computer(self, ori_enc_inps, ori_len_inps, beam_size):
        #print ("init_dec_state_computer!!!!")
        input_feed = {}
        input_feed[self.model.keep_prob.name] = 1.0
        for l in range(self.enc_len):
            input_feed[self.model.enc_inps[l].name] = ori_enc_inps[l]
        input_feed[self.model.enc_len_inps.name] = ori_len_inps

        output_feed = [self.model.dec_init_state, self.model.memory]
        [dec_init_state, memory] = self.sess.run(output_feed, input_feed)

        return dec_init_state, memory

    def dec_state_computer(self, inp, state, memory, ori_enc_inps, ori_len_inps, n_samples):
        input_feed = {}
        input_feed[self.model.keep_prob.name] = 1.0
        input_feed[self.model.dec_inps[0].name] = inp
        input_feed[self.model.beam_dec_state_c.name] = state[:, 0, :]
        input_feed[self.model.beam_dec_state_m.name] = state[:, 1, :]
        for l in range(self.enc_len):
            input_feed[self.model.enc_inps[l].name] = ori_enc_inps[l][0:n_samples]
        input_feed[self.model.enc_len_inps.name] = ori_len_inps[0:n_samples]
        input_feed[self.model.beam_memory.name] = memory[0:n_samples, :, :]

        output_feed = [self.model.next_out, self.model.next_state, self.model.align]
        [next_out, next_state, align] = self.sess.run(output_feed, input_feed)

        c = np.expand_dims(next_state[0], axis=1)
        m = np.expand_dims(next_state[1], axis=1)

        next_state = np.concatenate([c,m], axis=1)

        return next_out, next_state

if __name__ == "__main__":
    tf.app.run()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import Rewarder.codes.tfidf.graphs as graphs
from Rewarder.codes.tfidf.tool import Tool
from Rewarder.codes.tfidf.config import hps


''''construction for Evaluator'''
class BaseComputer(object):
    
    def __init__(self, graph, sep_device):
        self.__root_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../"
        # construct HParams
        self.hps = hps

        self.tool = Tool(sen_len=hps.sen_len)
        self.tool.load_dic (self.__root_dir + hps.vocab_path, self.__root_dir + hps.ivocab_path)

        vocab_size = self.tool.get_vocab_size()
        assert vocab_size > 0
        self.hps = self.hps._replace(vocab_size=vocab_size)
        self.hps = self.hps._replace(device=sep_device)

        self.sen_len = hps.sen_len

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.98)
        gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(graph=graph, 
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        self.model = graphs.SemanticModel(self.hps)
        self.model.build_eval_graph()
        self.load_model()


    def load_model(self):
        """load model."""
        saver = tf.train.Saver(tf.global_variables() , write_version=tf.train.SaverDef.V1)

        ckpt = tf.train.get_checkpoint_state(self.__root_dir+"model/tfidf/")
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" %
                    ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError("%s not found! " % ckpt.model_checkpoint_path)

        '''
        variables = tf.trainable_variables()
        total_parameters = 0
        for variable in variables:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("total_parameters: %d" % (total_parameters))
        input(">")
        '''

    def get_meaning_score(self, sens):
        batch_enc_inps, len_inps = self.tool.get_eval_batch_enc_sen(sens)
        scores = self.__preds_computer(batch_enc_inps, len_inps)

        return scores

# ------------------------
    def __preds_computer(self, ori_enc_inps, ori_len_inps):
        #print ("init_dec_state_computer!!!!")
        input_feed = {}
        input_feed[self.model.keep_prob.name] = 1.0
        for l in range(self.sen_len):
            input_feed[self.model.enc_inps[l].name] = ori_enc_inps[l]
        input_feed[self.model.enc_len_inps.name] = ori_len_inps


        output_feed = [self.model.preds]
        [preds] = self.sess.run(output_feed, input_feed)

        return preds

def main():
    pass



if __name__ == "__main__":
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import Rewarder.codes.tfidf.layers as layers_lib

class SemanticModel(object):
    def __init__(self, hps):
        # Create the model
        self.hps = hps
        self.sen_len = hps.sen_len
        self.device = self.hps.device
        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.Variable(float(self.hps.learning_rate), trainable=False)
        self.learning_rate_decay_op = \
            self.learning_rate.assign(self.learning_rate * self.hps.decay_rate)

        self.__build_placeholders()

        # Build modules and layers
        with tf.device(self.device):
            self.layers = {}

            self.layers['word_emb'] = layers_lib.Embedding(
                self.hps.vocab_size, self.hps.emb_size, name="word_emb")

            # Build Decoder
            self.layers['enc'] = layers_lib.BidirEncoder(
                self.hps.hidden_size, keep_prob=self.keep_prob, name="enc")

            self.layers['mlp_classifier'] = layers_lib.MLPLayer([self.hps.hidden_size, self.hps.emb_size, 64, 1], 
                activs=['leak_relu', 'leak_relu', 'leak_relu', None], keep_prob=self.keep_prob, name='mlp_classifier')

    def __build_placeholders(self):
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.enc_len_inps = tf.placeholder(tf.int32, shape=[None],  name="enc_len_inps")

        self.enc_inps = []
        for i in range(self.sen_len):
            self.enc_inps.append(tf.placeholder(tf.int32, shape=[None], name="enc_inps{0}".format(i)))

        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

    @property
    def pretrained_variables(self):
        variables = self.layers['word_emb'].trainable_weights +  self.layers['enc'].trainable_weights
        return variables

    def training(self):
        print ("using device: %s" % self.hps.device)
        with tf.device(self.hps.device):

            loss, preds = self.__build_graph()
            train_op, gradients, regular_loss = self.__optimization(loss, self.hps.l2_weight)
            
        return train_op, loss, regular_loss, preds, self.global_step

    def __build_graph(self):
        print (self.device)
        with tf.device(self.device):
            emb_enc_inps = [self.layers['word_emb'](x) for x in self.enc_inps]

            # build Encoder
            _, enc_fw, enc_bw = self.layers['enc'](emb_enc_inps, self.enc_len_inps)
            state = array_ops.concat([enc_fw, enc_bw], axis=1)

            preds = self.layers['mlp_classifier'](state)
            preds = tf.squeeze(preds, axis=1)
            #print (preds.get_shape())
            loss = tf.losses.huber_loss(
                self.labels, preds,
                weights=1.0, delta=1.0)

            #----------------------------------
            return loss, preds

    def __optimization(self, loss, l2_weight):
        params = tf.trainable_variables()                
        regularizers = []

        for param in params:
            name = param.name
            print (name)
            regularizers.append(tf.nn.l2_loss(param))
                
        regular_loss = math_ops.reduce_sum(regularizers)
        
        total_loss = loss + l2_weight * regular_loss
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients = tf.gradients(total_loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.hps.max_gradient_norm)
                
        train_op = opt.apply_gradients( zip(clipped_gradients, params), global_step=self.global_step)
        return train_op, gradients, regular_loss

    #-----------------------------------
    def build_eval_graph(self):
        print ("using device: %s" % self.hps.device)
        with tf.device(self.hps.device):
            emb_enc_inps = [self.layers['word_emb'](x) for x in self.enc_inps]

            # build Encoder
            _, enc_fw, enc_bw = self.layers['enc'](emb_enc_inps, self.enc_len_inps)
            state = array_ops.concat([enc_fw, enc_bw], axis=1)

            preds = self.layers['mlp_classifier'](state)
            self.preds = tf.squeeze(preds, axis=1)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import Rewarder.codes.lm.layers as layers_lib

class LMModel(object):
    """Language Model"""
    def __init__(self, hps):
        self.global_step = tf.train.get_or_create_global_step()
        self.hps = hps

        self.learning_rate = tf.Variable(float(self.hps.learning_rate), trainable=False)
        self.learning_rate_decay_op = \
            self.learning_rate.assign(self.learning_rate * self.hps.lr_decay)

        self.__build_placeholder()

        with tf.device(self.hps.device):
            self.layers = {}
            self.layers['emb'] = layers_lib.Embedding(
                self.hps.vocab_size, self.hps.emb_size, name="emb")

            self.layers['enc'] = layers_lib.Encoder(
                self.hps.hidden_size, self.hps.layer_num, self.keep_prob, name="enc")

            self.layers['out_layer'] = layers_lib.OutLayer(self.hps.vocab_size,
                self.hps.hidden_size, keep_prob=self.keep_prob, name='out_layer')


    def __build_placeholder(self):
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.inputs = tf.placeholder(tf.int32, [None, None], name = "inputs")
        self.targets = tf.placeholder(tf.int32, [None, None], name = "targets")
        self.len_inputs = tf.placeholder(tf.int32, shape=[None],  name="len_inputs")

    def __build_graph(self):
        with tf.device(self.hps.device):
            emb_inps = self.layers['emb'](self.inputs)

            # NOTE: the idx of PAD is 0
            inputs_mask = tf.sign(self.inputs, name="inputs_mask")
            sen_lens = tf.reduce_sum(inputs_mask, 1)

            targets_mask = tf.sign(self.targets, name="targets_mask")
            targets_mask = tf.cast(targets_mask, tf.float32)

            with tf.device(self.hps.device):
                #[batch_size, max_time, cell_fw.output_size]
                #print ("graph!!!!")
                states = self.layers['enc'](emb_inps, sen_lens)
                #print (states.get_shape())

                logits = self.layers['out_layer'](states)
                loss = layers_lib.sequence_loss(logits, self.targets, targets_mask)

                #next_prob = tf.nn.softmax(tf.matmul(tf.expand_dims(outputs[-1,-1,:], 0), softmax_w_prob) + softmax_b)
                #self._next_prob = next_prob
                self.logits = tf.nn.softmax(logits, axis=-1)

            return loss, sen_lens

    def build_eval_graph(self):
        with tf.device(self.hps.device):
            emb_inps = self.layers['emb'](self.inputs)

            # NOTE: the idx of PAD is 0
            inputs_mask = tf.sign(self.inputs, name="inputs_mask")
            sen_lens = tf.reduce_sum(inputs_mask, 1)

            targets_mask = tf.sign(self.targets, name="targets_mask")
            self.targets_mask = tf.cast(targets_mask, tf.float32)

            with tf.device(self.hps.device):
                #[batch_size, max_time, cell_fw.output_size]
                states = self.layers['enc'](emb_inps, sen_lens)

                logits = self.layers['out_layer'](states)
                #next_prob = tf.nn.softmax(tf.matmul(tf.expand_dims(outputs[-1,-1,:], 0), softmax_w_prob) + softmax_b)
                #self._next_prob = next_prob
                self.probs = tf.nn.softmax(logits, axis=-1)

    def training(self):
        loss, debug = self.__build_graph()
        train_op, _, regular_loss = self.__optimize(loss, self.global_step)
        return train_op, loss, regular_loss, self.global_step, debug

    def __optimize(self, loss, global_step_op):
        print ("optimize")
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        params = tf.trainable_variables()

        regularizers = []
        for param in params:
            name = param.name
            regularizers.append(tf.nn.l2_loss(param))
            print (name)
        print (len(params))

        regular_loss = math_ops.reduce_sum(regularizers)
        loss = loss + self.hps.l2_ratio*regular_loss

        gradients = tf.gradients(loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.hps.max_grad_norm)

        train_op = opt.apply_gradients( zip(clipped_gradients, params), global_step=global_step_op)
        gradients = clipped_gradients

        return train_op, gradients, regular_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import Rewarder.codes.mi.layers as layers_lib

class Seq2Seq(object):
    def __init__(self, hps, init_emb=None):
        # Create the model
        self.hps = hps
        self.enc_len = hps.enc_len
        self.dec_len = hps.dec_len
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
                self.hps.hidden_size, layers_num=self.hps.layers_num,
                keep_prob=self.keep_prob, name="enc")

            # The decoder cell
            self.layers['dec'] = layers_lib.Decoder(
                self.hps.hidden_size, self.keep_prob, name="dec")
            
            self.layers['dinit'] = layers_lib.MLPLayer([self.hps.hidden_size*2], activs=['tanh'],
                keep_prob=self.keep_prob, name='dinit')

            self.layers['attn_layer'] = layers_lib.AttentionLayer(self.hps.hidden_size*2, "attn_layer")

            self.layers['out_proj'] = layers_lib.MLPLayer([self.hps.hidden_size*4, self.hps.vocab_size], 
                activs=['tanh', None], keep_prob=self.keep_prob, name='out_proj')

            self.layers['dec_merge'] = layers_lib.MLPLayer([self.hps.hidden_size], 
                activs=[None], keep_prob=self.keep_prob, name='dec_merge')

            # loss
            self.layers['softmax_loss'] = layers_lib.SoftmaxLoss(with_probs=False, name='softmax_loss')

    def __build_placeholders(self):
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.enc_len_inps = tf.placeholder(tf.int32, shape=[None],  name="enc_len_inps")

        self.enc_inps = []
        for i in range(self.enc_len):
            self.enc_inps.append(tf.placeholder(tf.int32, shape=[None], name="enc_inps{0}".format(i)))

        self.dec_inps = []
        self.trg_weights = []
        # The dec_len is added 1 since we build the targets by shifting one position of the decoder inputs
        for i in range(self.dec_len + 1):
            self.dec_inps.append(tf.placeholder(tf.int32, shape=[None], name="dec_inps{0}".format(i)))
            self.trg_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))   

        self.targets = [self.dec_inps[i + 1] for i in range(len(self.dec_inps) - 1)]        
        
        # -----------------------------------------------
        # For beam search
        self.beam_dec_state_c = tf.placeholder(tf.float32, 
                shape=[None, self.hps.hidden_size], name="beam_dec_state_c")
        self.beam_dec_state_m = tf.placeholder(tf.float32, 
                shape=[None, self.hps.hidden_size], name="beam_dec_state_m")

        self.beam_memory = tf.placeholder(tf.float32, 
                shape=[None, self.hps.enc_len, self.hps.hidden_size*2], name="beam_memory")

    def training(self):
        print ("using device: %s" % self.hps.device)
        with tf.device(self.hps.device):

            normed_outs, gen_loss = self.__build_graph()
            train_op, gradients, regular_loss = self.__optimization(gen_loss, self.hps.l2_weight)
            
        return train_op, normed_outs, gen_loss, regular_loss, self.global_step

    def __build_graph(self):
        print (self.device)
        with tf.device(self.device):
            emb_enc_inps = [self.layers['word_emb'](x) for x in self.enc_inps]
            emb_dec_inps = [self.layers['word_emb'](x) for x in self.dec_inps]

            # build Encoder
            memory, enc_sfw, enc_sbw = self.layers['enc'](emb_enc_inps, self.enc_len_inps)


            init_state = self.layers['dinit']([enc_sfw[0], enc_sfw[1], enc_sbw[0], enc_sbw[1]])
            init_state = array_ops.reshape(init_state, [-1, 2, self.hps.hidden_size])
            init_states = tf.unstack(init_state, axis=1)

            state = tf.nn.rnn_cell.LSTMStateTuple(init_states[0], init_states[1])
            align_state = tf.zeros([tf.shape(init_states[0])[0], self.hps.enc_len], dtype=tf.float32)

            dec_inps = emb_dec_inps[0:self.dec_len]
            normed_dec_outs = []

            for i, inp in enumerate(dec_inps):
                query = array_ops.concat([state[0], state[1]], axis=1)
                attns, align, align_state = self.layers['attn_layer'](memory, self.enc_len_inps, query, align_state)

                x = self.layers['dec_merge']([inp, attns])
                cell_out, state = self.layers['dec'](state, x)
                out = self.layers['out_proj']([cell_out, inp, attns])

                normed_dec_outs.append(tf.identity(out))

            weights = self.trg_weights[: self.dec_len]

            # gen loss
            gen_loss = self.layers['softmax_loss'](normed_dec_outs, self.targets[: self.dec_len], weights)

            #----------------------------------
            return normed_dec_outs, gen_loss

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
            print ("using device: %s" % self.hps.device)
            emb_enc_inps = [self.layers['word_emb'](x) for x in self.enc_inps]
            self.memory, enc_sfw, enc_sbw = self.layers['enc'](emb_enc_inps, self.enc_len_inps)

            init_state = self.layers['dinit']([enc_sfw[0], enc_sfw[1], enc_sbw[0], enc_sbw[1]])
            self.dec_init_state = array_ops.reshape(init_state, [-1, 2, self.hps.hidden_size])

            # decoder
            query = array_ops.concat([self.beam_dec_state_c, self.beam_dec_state_m], axis=1)
            align_state = tf.zeros([tf.shape(self.beam_dec_state_c)[0], self.hps.enc_len], dtype=tf.float32)
            attns, self.align, align_state = self.layers['attn_layer'](self.beam_memory, self.enc_len_inps, query, align_state)

            emb_dec_inp = self.layers['word_emb'](self.dec_inps[0])
            x = self.layers['dec_merge']([emb_dec_inp, attns])
            dec_state = tf.nn.rnn_cell.LSTMStateTuple(self.beam_dec_state_c, self.beam_dec_state_m)
            
            cell_out, self.next_state = self.layers['dec'](dec_state, x)
            out = self.layers['out_proj']([cell_out, emb_dec_inp, attns])

            self.next_out = tf.nn.softmax(out, axis=-1)
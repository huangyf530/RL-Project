from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



# Dependency imports


import tensorflow as tf
import Rewarder.codes.dis.adversarial_losses as adv_lib
import Rewarder.codes.dis.layers as layers_lib
import Rewarder.codes.dis.tool as tool

from Rewarder.codes.dis.config import FLAGS

class VatxtModel(object):
    def __init__(self, vocab_freq_path, vocab_size, EOS_ID, device):
        self.vocab_size = vocab_size
        self.EOS_ID = EOS_ID
        self.maxlen = 30
        self.device = device
        self.global_step = tf.train.get_or_create_global_step()
        self.vocab_freqs = tool.get_vocab_freqs(vocab_freq_path, vocab_size)
        
        self.__build_placeholder()

        # Cache intermediate Tensors that are reused
        self.tensors = {}

        with tf.device(self.device):
            # Construct layers which are reused in constructing the LM and
            # Classification graphs. Instantiating them all once here ensures that
            # variable reuse works correctly.
            self.layers = {}
            self.layers['embedding'] = layers_lib.Embedding(
                self.vocab_size, FLAGS.embedding_dims, self.vocab_freqs, FLAGS.keep_prob_emb)
            
            self.layers['gru'] = layers_lib.GRU(
                FLAGS.rnn_cell_size, FLAGS.rnn_num_layers, FLAGS.keep_prob_lstm_out,
                name='GRU')
            self.layers['gru_reverse'] = layers_lib.GRU(
                FLAGS.rnn_cell_size, FLAGS.rnn_num_layers, FLAGS.keep_prob_lstm_out, 
                name='GRU_Reverse')
            
            self.layers['lm_loss'] = layers_lib.SoftmaxLoss(
                self.vocab_size,
                name='LM_loss')
            self.layers['lm_loss_reverse'] = layers_lib.SoftmaxLoss(
                self.vocab_size,
                name='LM_loss_reverse')

            # Note that we use bi-directional rnn
            cl_logits_input_dim = FLAGS.rnn_cell_size * 2
            self.layers['cl_logits'] = layers_lib.cl_logits_subgraph(
                [256, 128, 64], cl_logits_input_dim,
                FLAGS.num_classes, FLAGS.keep_prob_cl_hidden)


    def __build_placeholder(self):
        
        # placeholder for lm
        self.lm_inputs_f = {}
        self.lm_inputs_f['x'] = tf.placeholder(tf.int32, [None, None], name = "lm_inputs_x")
        self.lm_inputs_f['labels'] = tf.placeholder(tf.int32, [None, None], name = "lm_inputs_labels")
        self.lm_inputs_f['weights'] = tf.placeholder(tf.float32, [None, None], name = "lm_inputs_weights")
        self.lm_inputs_f['length'] = tf.placeholder(tf.int32, [None], name = "lm_inputs_length")

        self.lm_inputs_r = {}
        self.lm_inputs_r['x'] = tf.placeholder(tf.int32, [None, None], name = "lm_inputs_x_i")
        self.lm_inputs_r['labels'] = tf.placeholder(tf.int32, [None, None], name = "lm_inputs_labels_i")
        self.lm_inputs_r['weights'] = tf.placeholder(tf.float32, [None, None], name = "lm_inputs_weights_i")
        self.lm_inputs_r['length'] = tf.identity(self.lm_inputs_f['length'])

        self.lm_inputs_f['eos_weights'] = tf.cast(tf.equal(self.lm_inputs_f['x'], self.EOS_ID), tf.float32)
        self.lm_inputs_r['eos_weights'] = tf.cast(tf.equal(self.lm_inputs_r['x'], self.EOS_ID), tf.float32)

        # placeholder for cl
        self.cl_inputs_f = {}
        self.cl_inputs_f['x'] = tf.placeholder(tf.int32, [None, None], name = "cl_inputs_x")
        self.cl_inputs_f['labels'] = tf.placeholder(tf.int32, [None, None], name = "cl_inputs_labels")
        self.cl_inputs_f['weights'] = tf.placeholder(tf.float32, [None, None], name = "cl_inputs_weights")
        self.cl_inputs_f['length'] = tf.placeholder(tf.int32, [None], name = "cl_inputs_length")

        self.cl_inputs_r = {}
        self.cl_inputs_r['x'] = tf.placeholder(tf.int32, [None, None], name = "cl_inputs_x_i")
        self.cl_inputs_r['labels'] = tf.identity(self.cl_inputs_f['labels'])
        self.cl_inputs_r['weights'] = tf.identity(self.cl_inputs_f['weights'])
        self.cl_inputs_r['length'] = tf.identity(self.cl_inputs_f['length'])


    @property
    def pretrained_variables(self):
        
        variables = (self.layers['embedding'].trainable_weights +
            self.layers['gru'].trainable_weights)
        variables.extend(self.layers['gru_reverse'].trainable_weights)
        return variables

    def classifier_training(self):
        loss, acc, pred = self.classifier_graph()
        train_op = optimize(loss, self.global_step)
        return train_op, loss, self.global_step, acc, pred

    def language_model_training(self):
        loss = self.language_model_graph()
        train_op = optimize(loss, self.global_step)
        return train_op, loss, self.global_step

    '''
    Constructs forward and reverse LM graphs from inputs to LM losses.
    * Caches the VatxtInput objects in `self.lm_inputs`
    * Caches tensors: `lm_embedded`, `lm_embedded_reverse`
    Args:
      compute_loss: bool, whether to compute and return the loss or stop after
        the LSTM computation.
    Returns:
      loss: scalar float, sum of forward and reverse losses.
    '''
    def language_model_graph(self, compute_loss=True):
        with tf.device(self.device):
            self.lm_inputs = self.lm_inputs_f, self.lm_inputs_r
            f_loss = self._lm_loss(self.lm_inputs_f, compute_loss=compute_loss)
            r_loss = self._lm_loss(
                self.lm_inputs_r,
                emb_key='lm_embedded_reverse',
                gru_layer='gru_reverse',
                lm_loss_layer='lm_loss_reverse',
                loss_name='lm_loss_reverse',
                compute_loss=compute_loss)
            if compute_loss:
                return f_loss + r_loss

    def _lm_loss(self, inputs,
               emb_key='lm_embedded',
               gru_layer='gru',
               lm_loss_layer='lm_loss',
               loss_name='lm_loss',
               compute_loss=True):
        embedded = self.layers['embedding'](inputs['x'])
        self.tensors[emb_key] = embedded
        gru_out, next_state = self.layers[gru_layer](embedded, inputs['length'])
        if compute_loss:
            loss = self.layers[lm_loss_layer](
                [gru_out, inputs['labels'], inputs['weights']])
            loss = tf.identity(loss)
            tf.summary.scalar(loss_name, loss)

            return loss

    '''
    Constructs classifier graph from inputs to classifier loss.
    Caches the VatxtInput objects in `self.cl_inputs`
    Caches tensors: `cl_embedded` (tuple of forward and reverse), `cl_logits`, `cl_loss`
    Returns:
      loss: scalar float.
    '''
    def classifier_graph(self):
        with tf.device(self.device):
            self.cl_inputs = self.cl_inputs_f, self.cl_inputs_r
            f_inputs, r_inputs = self.cl_inputs_f, self.cl_inputs_r

            # Embed both forward and reverse with a shared embedding
            embedded = [self.layers['embedding'](f_inputs['x']),
                self.layers['embedding'](r_inputs['x'])]
            
            self.tensors['cl_embedded'] = embedded

            _, next_states, logits, loss = self.cl_loss_from_embedding(
                embedded, return_intermediates=True)
            tf.summary.scalar('classification_loss', loss)
            self.tensors['cl_logits'] = logits
            self.tensors['cl_loss'] = loss

            acc, pred = layers_lib.accuracy(logits, f_inputs['labels'], f_inputs['weights'])
            indices = tf.stack([tf.range(FLAGS.batch_size), f_inputs['length'] - 1], 1)
            pred = tf.gather_nd(pred, indices)
            tf.summary.scalar('accuracy', acc)

            adv_loss = (self.adversarial_loss() * tf.constant(
                FLAGS.adv_reg_coeff, name='adv_reg_coeff'))
            tf.summary.scalar('adversarial_loss', adv_loss)

            total_loss = loss + adv_loss
            total_loss = tf.identity(total_loss)
            tf.summary.scalar('total_classification_loss', total_loss)
            return total_loss, acc, pred

    '''
    Constructs classifier evaluation graph.
    '''
    def eval_graph(self):
        inputs = self.cl_inputs_f, self.cl_inputs_r
        f_inputs, r_inputs = inputs
        embedded = [self.layers['embedding'](f_inputs['x']),
            self.layers['embedding'](r_inputs['x'])]
        logits= self.cl_loss_from_embedding(
                embedded, inputs=inputs, only_logits=True)

        pred = layers_lib.predictions(logits)
        pred_prob = tf.nn.sigmoid(logits)
        #print ("lalala")
        #print (pred.get_shape())
        #print (pred_prob.get_shape())
        batch_size = tf.shape(pred)[0]
        indices = tf.stack([tf.range(batch_size), f_inputs['length'] - 1], 1)
        #print(indices.get_shape())
        pred = tf.gather_nd(pred, indices)

        pred_prob = tf.gather_nd(pred_prob, indices)

        return pred, pred_prob


    def cl_loss_from_embedding(self, embedded,
        inputs=None, return_intermediates=False, only_logits=False):

        if inputs is None:
            inputs = self.cl_inputs

        out = []
        for (layer_name, emb, inp) in zip(['gru', 'gru_reverse'], embedded, inputs):
            out.append(self.layers[layer_name](emb, inp['length']))

        gru_outs, next_states = zip(*out)
        # Concatenate output of forward and reverse RNNs
        # batch_size, max_time, cell.output_size
        gru_out = tf.concat(gru_outs, 2)

        #logits: 2-D [timesteps*batch_size, m] float tensor, where m=1 if
        #num_classes=2, otherwise m=num_classes.
        logits = self.layers['cl_logits'](gru_out)

        #print (logits.get_shape())

        if only_logits:
            return logits

        f_inputs, _ = inputs  # pylint: disable=unpacking-non-sequence
        loss = layers_lib.classification_loss(logits, f_inputs['labels'], f_inputs['weights'])

        if return_intermediates:
            return gru_out, next_states, logits, loss
        else:
            return loss
    '''
    Compute adversarial loss based on FLAGS.adv_training_method.
    '''
    def adversarial_loss(self):
        def adversarial_loss():
            return adv_lib.adversarial_loss_bidir(self.tensors['cl_embedded'],
                self.tensors['cl_loss'],
                self.cl_loss_from_embedding)
        '''
        Computes virtual adversarial loss.
        Uses lm_inputs and constructs the language model graph if it hasn't yet
            been constructed.
        Also ensures that the LM input states are saved for LSTM state-saving
            BPTT.
        Returns:
            loss: float scalar.
        '''
        def virtual_adversarial_loss():
            self.language_model_graph(compute_loss=False)

            def logits_from_embedding(embedded, return_next_state=False):
                _, next_states, logits, _ = self.cl_loss_from_embedding(
                    embedded, inputs=self.lm_inputs, return_intermediates=True)
                if return_next_state:
                    return next_states, logits
                else:
                    return logits

            lm_embedded = (self.tensors['lm_embedded'],
                self.tensors['lm_embedded_reverse'])
            next_states, lm_cl_logits = logits_from_embedding(
                lm_embedded, return_next_state=True)

            va_loss = adv_lib.virtual_adversarial_loss_bidir(
                lm_cl_logits, lm_embedded, self.lm_inputs, logits_from_embedding, self.maxlen)

            '''
            saves = [ inp.save_state(state)
                for (inp, state) in zip(self.lm_inputs, next_states)
            ]
            with tf.control_dependencies(saves):
                va_loss = tf.identity(va_loss)
            '''

            va_loss = tf.identity(va_loss)
            return va_loss

        # end  of virtual_adversarial_loss
        #return adversarial_loss()
        return adversarial_loss() + virtual_adversarial_loss()
        #return virtual_adversarial_loss()

def optimize(loss, global_step):
    return layers_lib.optimize(
        loss, global_step, FLAGS.max_grad_norm, FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor)

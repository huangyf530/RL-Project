from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical

class Embedding(object):
    """
    Embedding layer
    """
    def __init__(self,
               vocab_size, emb_size,
               init_emb = None, name = 'embedding',
               trainable=True):

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.name = name
        self.trainable = trainable
        self.reuse = None
        self.trainable_weights = None
        self.init_emb = init_emb

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            if self.init_emb is not None:
                initializer = tf.constant_initializer(self.init_emb)
                if not self.reuse:
                    print ("Initialize embedding with pre-trained vectors.")
            else:
                initializer = tf.truncated_normal_initializer(stddev=1e-4)
                if not self.reuse:
                    print ("Initialize embedding with normal distribution.")

            word_emb = tf.get_variable('word_emb', [self.vocab_size, self.emb_size], 
                dtype=tf.float32, initializer=initializer, trainable= self.trainable)

            embedded = tf.nn.embedding_lookup(word_emb, x)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return embedded

class BidirEncoder(object):
    """
    Bidirectional Encoder
    Exposes variables in `trainable_weights` property.
    """
    def __init__(self, cell_size, layers_num, keep_prob=1.0, name='bidirenc'):
        self.cell_size = cell_size
        self.layers_num = layers_num
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, seq_lens):
        #print ("Encoder!!!!!")
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            def get_a_cell(name):
                cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse, name=name)
                cell =  tf.nn.rnn_cell.DropoutWrapper(cell, 
                    output_keep_prob=self.keep_prob,
                    input_keep_prob = self.keep_prob)
                return cell

            multi_cell_fw = tf.nn.rnn_cell.MultiRNNCell([get_a_cell('cell_fw'+str(i)) 
                for i in range(self.layers_num)], state_is_tuple=True)

            multi_cell_bw = tf.nn.rnn_cell.MultiRNNCell([get_a_cell('cell_bw'+str(i)) 
                for i in range(self.layers_num)], state_is_tuple=True)

            sequence = []
            for emb in x:
                inp = tf.expand_dims(emb, axis=1)
                sequence.append(inp)
            sequence = array_ops.concat(sequence, axis=1)
            outs , (state_fw, state_bw)  = tf.nn.bidirectional_dynamic_rnn(
                    multi_cell_fw, multi_cell_bw, sequence, sequence_length=seq_lens, dtype=tf.float32, time_major=False)


            concat_outs = array_ops.concat([outs[0], outs[1]], axis=-1)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return concat_outs, state_fw[-1], state_bw[-1]

class Decoder(object):
    '''
    Decoder with an output projection layer
    Exposes variables in `trainable_weights` property.
    '''
    def __init__(self, cell_size, keep_prob=1.0, name='dec'):
        self.cell_size = cell_size
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, state, x):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse)
            cell =  tf.nn.rnn_cell.DropoutWrapper(cell, 
                    output_keep_prob=self.keep_prob,
                    input_keep_prob = self.keep_prob)

            cell_out, next_state = cell(x, state)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return cell_out, next_state

class AttentionLayer(object):
    '''
    The attention layer
    Exposes variables in `trainable_weights` property.
    '''
    def __init__(self, units_num, name='attention'):
        self.units_num = units_num
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, attentions, attns_lens, query, align_state):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            attention_layer = tf.contrib.seq2seq.BahdanauAttention(self.units_num,
                    attentions, attns_lens, True)
            align, align_state = attention_layer(query, align_state)
            attns = tf.multiply(attentions, tf.expand_dims(align, axis=-1))
            attns = math_ops.reduce_sum(attns, axis=1)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return attns, align, align_state

class MLPLayer(object):
    '''
    MLP layer
    Exposes variables in `trainable_weights` property.
    '''
    def __init__(self, out_sizes, activs=None, keep_prob=1.0, trainable=True, name='mlp'):
        
        self.out_sizes = out_sizes
        if activs is None:
            activs = ['tanh'] * len(out_sizes)
        self.activs = activs
        self.keep_prob = keep_prob
        self.trainable = trainable
        self.name = name
        self.reuse = None
        self.trainable_weights = None

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:

            out = x
            layers_num = len(self.out_sizes)
            for i, (out_size, activ) in enumerate(zip(self.out_sizes, self.activs)):
                out = linear(out, out_size, bias=True, scope="mlp"+str(i), trainable=self.trainable)
                assert activ == 'tanh' or activ == 'relu' or activ == 'leak_relu' or activ is None
                if activ == 'tanh':
                    out =  tf.tanh(out)
                elif activ == 'relu':
                    out =  tf.nn.relu(out)
                elif activ == 'leak_relu':
                    LeakyReLU = tf.keras.layers.LeakyReLU(0.2)
                    out = LeakyReLU(out)

                if layers_num > 1 and i < layers_num-1:
                    out = tf.nn.dropout(out, self.keep_prob)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return out

# ----------------------------------------
# Tool functions
class SoftmaxLoss(object):
    '''Softmax xentropy'''
    def __init__(self, name, with_probs=False,
        average_across_timesteps=True,
        average_across_batch=True,
        softmax_loss_function=None):
        self.name = name
        self.with_probs = with_probs
        self.average_across_timesteps = average_across_timesteps
        self.average_across_batch = average_across_batch
        self.softmax_loss_function = softmax_loss_function

    def __call__(self, logits, targets, weights):
        
        with ops.name_scope(self.name):
            if self.with_probs:
                cost = math_ops.reduce_sum(sequence_loss_by_example_with_probs(
                    logits, targets, weights,
                    average_across_timesteps=self.average_across_timesteps))
            else:
                cost = math_ops.reduce_sum(sequence_loss_by_example(
                    logits, targets, weights,
                    average_across_timesteps=self.average_across_timesteps,
                    softmax_loss_function=self.softmax_loss_function))

            if self.average_across_batch:
                batch_size = array_ops.shape(targets[0])[0]
                return cost / math_ops.cast(batch_size, cost.dtype)
            else:
                return cost

def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """
    Weighted cross-entropy loss for a sequence of logits (per example).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
            "%d, %d, %d." % (len(logits), len(weights), len(targets)))

    with ops.name_scope("sequence_loss_by_example"):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)

        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size

    return log_perps

def sequence_loss_by_example_with_probs(probability, targets, weights,
                             average_across_timesteps=True):
    """
    Weighted cross-entropy loss for a sequence of logits (per example).
    """
    #print ("sequence_loss_by_example_with_probs!!!")
    if len(targets) != len(probability) or len(weights) != len(probability):
        raise ValueError("Lengths of probability, weights, and targets must be the same "
            "%d, %d, %d." % (len(probability), len(weights), len(targets)))

    with ops.name_scope("sequence_loss_by_example_with_probs"):
        log_perp_list = []
        batch_size = array_ops.shape(targets[0])[0]
        batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)

        for prob, target, weight in zip(probability, targets, weights):
            #print ("lalala")
            #print (prob.get_shape())
            indices = tf.stack((batch_nums, target), axis=1) # shape (batch_size, which target)
            #print(indices.get_shape())
            crossent = tf.gather_nd(prob, indices)
            #print(crossent.get_shape())
            crossent = -math_ops.log(crossent+1e-12)
            #print(crossent.get_shape())
            log_perp_list.append(crossent * weight)

        log_perps = math_ops.add_n(log_perp_list)
        #print(log_perps.get_shape())
        if average_across_timesteps:
            total_size = math_ops.add_n(weights) + 1e-12 # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size

    return log_perps


def linear(inputs, output_size, bias=True, concat=True, trainable=True, dtype=None, scope=None):
    """
    Linear layer. The code of this funciton is from the THUMT project
    :param inputs: A Tensor or a list of Tensors with shape [batch, input_size]
    :param output_size: An integer specify the output size
    :param bias: a boolean value indicate whether to use bias term
    :param concat: a boolean value indicate whether to concatenate all inputs
    :param dtype: an instance of tf.DType, the default value is ``tf.float32''
    :param scope: the scope of this layer, the default value is ``linear''
    :returns: a Tensor with shape [batch, output_size]
    :raises RuntimeError: raises ``RuntimeError'' when input sizes do not
                          compatible with each other
    """

    with tf.variable_scope(scope, values=[inputs]):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)
            shape = [input_size, output_size]
            matrix = tf.get_variable("Matrix", shape, dtype=dtype, trainable=trainable)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype, trainable=trainable)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.get_variable("Bias", shape, dtype=dtype, trainable=trainable)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)

        return output
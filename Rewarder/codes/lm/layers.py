# Dependency imports
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn

class Embedding(object):
    """
    Embedding layer
    """
    def __init__(self,
               vocab_size,
               emb_size,
               init_emb=None,
               name='embedding',
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
                print ("Initialize embedding with pre-trained matrix.")
                initializer = tf.constant_initializer(self.init_emb)
            else:
                print ("Initialize embedding with normal distribution.")
                initializer = tf.truncated_normal_initializer(stddev=1e-4)

            word_emb = tf.get_variable('word_emb', [self.vocab_size, self.emb_size], 
                dtype=tf.float32, initializer=initializer, trainable= self.trainable)

            embedded = tf.nn.embedding_lookup(word_emb, x)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return embedded

'''
class BidirEncoder(object):
    """
    Bidirectional Encoder
    Exposes variables in `trainable_weights` property.
    """
    def __init__(self, cell_size, layer_num=1, keep_prob=1.0, name='bidirenc'):
        self.cell_size = cell_size
        self.layer_num = layer_num
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, seq_lens):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:

            def get_a_cell(name):
                cell = tf.nn.rnn_cell.GRUCell(self.cell_size, 
                    reuse=tf.get_variable_scope().reuse, name=name)
                cell =  tf.nn.rnn_cell.DropoutWrapper(cell, 
                    output_keep_prob=self.keep_prob,
                    input_keep_prob = self.keep_prob)
                return cell

            multi_cell_fw = tf.nn.rnn_cell.MultiRNNCell([get_a_cell('fw'+str(i)) for i in range(self.layer_num)], state_is_tuple=True)
            multi_cell_bw = tf.nn.rnn_cell.MultiRNNCell([get_a_cell('bw'+str(i)) for i in range(self.layer_num)], state_is_tuple=True)

            outs , (state_fw, state_bw)  = tf.nn.bidirectional_dynamic_rnn(
                multi_cell_fw, multi_cell_bw, x, sequence_length=seq_lens, dtype=tf.float32)

            fin_states = array_ops.concat([outs[0], outs[1]], axis=2)
            #print ("Encoder!!!!")
            #print (fin_states.get_shape())

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return fin_states
'''

class Encoder(object):
    """
    Bidirectional Encoder
    Exposes variables in `trainable_weights` property.
    """
    def __init__(self, cell_size, layer_num=1, keep_prob=1.0, name='bidirenc'):
        self.cell_size = cell_size
        self.layer_num = layer_num
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, seq_lens):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:

            def get_a_cell(name):
                cell = tf.nn.rnn_cell.GRUCell(self.cell_size, 
                    reuse=tf.get_variable_scope().reuse, name=name)
                cell =  tf.nn.rnn_cell.DropoutWrapper(cell, 
                    output_keep_prob=self.keep_prob,
                    input_keep_prob = self.keep_prob)
                return cell

            multi_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell('cell'+str(i)) for i in range(self.layer_num)], state_is_tuple=True)

            outs , state  = tf.nn.dynamic_rnn(
                multi_cell, x, sequence_length=seq_lens, dtype=tf.float32)

            #print ("Encoder!!!!")
            #print (outs.get_shape())

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return outs

class OutLayer(object):
    # an output layer with convolution
    def __init__(self, vocab_size, input_size, keep_prob=1.0, name='outlayer'):
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.keep_prob = keep_prob
        self.name = name
        self.reuse = None
        self.trainable_weights = None

    def __call__(self, x):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            #print ("outlayer!!!")
            # x: [batch_size, max_time, cell_fw.output_size]
            w1 = tf.get_variable("w1", [1, 1, self.input_size, self.input_size*2], dtype=tf.float32)
            b1 = tf.get_variable("b1", [self.input_size*2], dtype=tf.float32)
            
            h1 = tf.expand_dims(x, 2)
            h1_features = tf.nn.conv2d(h1, w1, [1, 1, 1, 1], "SAME")
            h1_features = tf.tanh(h1_features + b1)

            h1_features = tf.nn.dropout(h1_features, self.keep_prob)
            #print (h1_features.get_shape())

            w2 = tf.get_variable("w2", [1, 1, self.input_size*2, self.vocab_size], dtype=tf.float32)
            b2 = tf.get_variable("b2", [self.vocab_size], dtype=tf.float32)
            h2_features = tf.nn.conv2d(h1_features, w2, [1, 1, 1, 1], "SAME") + b2

            #print (h2_features.get_shape())
            logits = tf.squeeze(h2_features, [2])
            #print (logits.get_shape())

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return logits

class MLPLayer(object):
    '''
    MLP layer
    Exposes variables in `trainable_weights` property.
    '''
    def __init__(self, out_sizes, activs=None, keep_prob=1., trainable=True, name='mlp'):
        
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
            layers_num = len(self.out_sizes)
            out = x
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

class SoftmaxLoss(object):
    '''Softmax xentropy'''
    def __init__(self, vocab_size, name):
        self.vocab_size = vocab_size
        self.name = name

    def __call__(self, logits, targets, weights,
        average_across_timesteps=True, average_across_batch=True,
        softmax_loss_function=None, name=None):
        
        with ops.name_scope(self.name):
            cost = math_ops.reduce_sum(sequence_loss_by_example(
                logits, targets, weights,
                average_across_timesteps=average_across_timesteps,
                softmax_loss_function=softmax_loss_function))

            if average_across_batch:
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


#---------------------------------------
def linear(inputs, output_size, bias=True, concat=True, trainable=True, dtype=None, scope=None):
    """
    Linear layer. The code of this funciton is form THUMT
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


def sequence_loss(logits, targets, weights,
    average_across_timesteps=True, average_across_batch=True,
    softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).
        Args:
            logits: A 3D Tensor of shape
                [batch_size x sequence_length x num_decoder_symbols] and dtype float.
                The logits correspond to the prediction across all classes at each timestep.
    
            targets: A 2D Tensor of shape [batch_size x sequence_length] and dtype int. 
                The target represents the true class at each timestep.
            
            weights: A 2D Tensor of shape [batch_size x sequence_length] and dtype
                float. Weights constitutes the weighting of each prediction in the
                sequence. When using weights as masking set all valid timesteps to 1 and
                all padded timesteps to 0.
            
            average_across_timesteps: If set, sum the cost across the sequence
                dimension and divide the cost by the total label weight across timesteps.
            
            average_across_batch: If set, sum the cost across the batch dimension and
                divide the returned cost by the batch size.
            
            softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
                to be used instead of the standard softmax (the default if this is None).
            
            name: Optional name for this operation, defaults to "sequence_loss".
        Returns:
            A scalar float Tensor: The average log-perplexity per symbol (weighted).

        Raises:
            ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
    """
    if len(logits.get_shape()) != 3:
        raise ValueError("Logits must be a "
                     "[batch_size x sequence_length x logits] tensor")
    if len(targets.get_shape()) != 2:
        raise ValueError("Targets must be a [batch_size x sequence_length] "
                     "tensor")
    if len(weights.get_shape()) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] "
                     "tensor")
    with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
        num_classes = array_ops.shape(logits)[2]
        probs_flat = array_ops.reshape(logits, [-1, num_classes])
        targets = array_ops.reshape(targets, [-1])
        if softmax_loss_function is None:
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=probs_flat)
        else:
            crossent = softmax_loss_function(targets, probs_flat)
        
        crossent = crossent * array_ops.reshape(weights, [-1])
        if average_across_timesteps and average_across_batch:
            crossent = math_ops.reduce_sum(crossent)
            total_size = math_ops.reduce_sum(weights)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        else:
            batch_size = array_ops.shape(logits)[0]
            sequence_length = array_ops.shape(logits)[1]
            crossent = array_ops.reshape(crossent, [batch_size, sequence_length])
        if average_across_timesteps and not average_across_batch:
            crossent = math_ops.reduce_sum(crossent, axis=[1])
            total_size = math_ops.reduce_sum(weights, axis=[1])
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        if not average_across_timesteps and average_across_batch:
            crossent = math_ops.reduce_sum(crossent, axis=[0])
            total_size = math_ops.reduce_sum(weights, axis=[0])
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            crossent /= total_size
        return crossent
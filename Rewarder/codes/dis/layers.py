"""Layers for VatxtModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
K = tf.keras

'''
Embedding layer with frequency-based normalization and dropout.
'''
class Embedding(K.layers.Layer):
    def __init__(self,
               vocab_size,
               emb_dim,
               vocab_freqs,
               keep_prob=1.0,
               **kwargs):

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.keep_prob = keep_prob

        self.vocab_freqs = tf.constant(
            vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))

        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.device('/cpu:0'):
            self.var = self.add_weight(
                shape=(self.vocab_size, self.emb_dim),
                initializer= tf.random_uniform_initializer(minval=-1.0, maxval=1.0, dtype=tf.float32),
                name='embedding', dtype=tf.float32)

        self.var = self._normalize(self.var)
        super(Embedding, self).build(input_shape)

    def _normalize(self, emb):
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev

    def call(self, x):
        embedded = tf.nn.embedding_lookup(self.var, x)
        if self.keep_prob < 1.:

            # Use same dropout masks at each timestep with specifying noise_shape.
            # This slightly improves performance.
            # Please see https://arxiv.org/abs/1512.05287 for the theoretical
            # explanation.
            embedded = tf.nn.dropout(
                embedded, self.keep_prob)
        return embedded

'''
LSTM layer using dynamic_rnn.
Exposes variables in `trainable_weights` property.
'''
class GRU(object):
    def __init__(self, cell_size, num_layers=1, keep_prob=1., name='GRU'):
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, seq_length):
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
        
            cell = tf.contrib.rnn.MultiRNNCell([
                tf.nn.rnn_cell.GRUCell(
                    self.cell_size,
                    reuse=tf.get_variable_scope().reuse)
                for _ in range(self.num_layers)], state_is_tuple=False)
      
            #cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
            #print ("GRU!!!!")
            #print (self.num_layers)


            # shape(x) = (batch_size, num_timesteps, embedding_dim)
            out, next_state = tf.nn.dynamic_rnn(
                cell, x, sequence_length=seq_length, dtype=tf.float32)

            #print (type(out))
            #print (out.get_shape())

            #print (type(next_state))
            #print (len(next_state))
            #print (next_state[0].get_shape())
            #print (next_state[1].get_shape())

            # shape(out) = (batch_size, timesteps, cell_size)
            if self.keep_prob < 1.0:
                out = tf.nn.dropout(out, self.keep_prob)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return out, next_state

'''Softmax xentropy'''
class SoftmaxLoss(K.layers.Layer):
    def __init__(self,
        vocab_size,
        **kwargs):

        self.vocab_size = vocab_size
        super(SoftmaxLoss, self).__init__(**kwargs)
        # The output projection layer
        self.multiclass_dense_layer = K.layers.Dense(self.vocab_size)

    def build(self, input_shape):
        input_shape = input_shape[0]
        self.multiclass_dense_layer.build(input_shape)
        super(SoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        x, labels, weights = inputs
        logits = self.multiclass_dense_layer(x)
        lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
        lm_loss = tf.identity(
            tf.reduce_sum(lm_loss * weights) / _num_labels(weights),
            name='lm_xentropy_loss')

        return lm_loss

'''
Construct multiple ReLU layers with dropout and a linear layer.
'''
def cl_logits_subgraph(layer_sizes, input_size, num_classes, keep_prob=1.):
    subgraph = K.models.Sequential(name='cl_logits')
    for i, layer_size in enumerate(layer_sizes):
        if i == 0:
            subgraph.add(
                K.layers.Dense(layer_size, activation='relu', input_dim=input_size))
        else:
            subgraph.add(K.layers.Dense(layer_size, activation='relu'))

    if keep_prob < 1.:
        subgraph.add(K.layers.Dropout(1. - keep_prob))
    subgraph.add(K.layers.Dense(1 if num_classes == 2 else num_classes))
    return subgraph

'''
Computes cross entropy loss between logits and labels.
Args:
    logits: 2-D [timesteps*batch_size, m] float tensor, where m=1 if
      num_classes=2, otherwise m=num_classes.
    labels: 1-D [timesteps*batch_size] integer tensor.
    weights: 1-D [timesteps*batch_size] float tensor.
Returns:
    Loss scalar of type float.
'''
def classification_loss(logits, labels, weights):
    inner_dim = logits.get_shape().as_list()[-1]
    with tf.name_scope('classifier_loss'):
        # Logistic loss
        if inner_dim == 1:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.squeeze(logits, -1), labels=tf.cast(labels, tf.float32))
        # Softmax loss
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)

        num_lab = _num_labels(weights)
        tf.summary.scalar('num_labels', num_lab)
        return tf.identity(
            tf.reduce_sum(weights * loss) / num_lab, name='classification_xentropy')
'''
Computes prediction accuracy.
Args:
    logits: 2-D classifier logits [timesteps*batch_size, num_classes]
    targets: 1-D [timesteps*batch_size] integer tensor.
    weights: 1-D [timesteps*batch_size] float tensor
Returns:
    Accuracy: float scalar.
'''
def accuracy(logits, targets, weights):
    with tf.name_scope('accuracy'):
        pred = predictions(logits)
        eq = tf.cast(tf.equal(pred, targets), tf.float32)
        return tf.identity(
            tf.reduce_sum(weights * eq) / _num_labels(weights), name='accuracy'), pred

'''Class prediction from logits.'''
def predictions(logits):
    inner_dim = logits.get_shape().as_list()[-1]
    with tf.name_scope('predictions'):
        if inner_dim == 1:
            # For binary classification
            pred = tf.cast(tf.greater(tf.squeeze(logits, -1), 0.), tf.int32)
        else:
            # For multi-class classification
            pred = tf.argmax(logits, axis=2, output_type=tf.int32)
        return pred

def _num_labels(weights):
    """Number of 1's in weights. Returns 1. if 0."""
    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)
    return num_labels

"""Builds optimization graph.

  * Creates an optimizer, and optionally wraps with SyncReplicasOptimizer
  * Computes, clips, and applies gradients
  * Maintains moving averages for all trainable variables
  * Summarizes variables and gradients

  Args:
    loss: scalar loss to minimize.
    global_step: integer scalar Variable.
    max_grad_norm: float scalar. Grads will be clipped to this value.
    lr: float scalar, learning rate.
    lr_decay: float scalar, learning rate decay rate.
    sync_replicas: bool, whether to use SyncReplicasOptimizer.
    replicas_to_aggregate: int, number of replicas to aggregate when using
      SyncReplicasOptimizer.
    task_id: int, id of the current task; used to ensure proper initialization
      of SyncReplicasOptimizer.

  Returns:
    train_op
"""
def optimize(loss, 
    global_step,
    max_grad_norm,
    lr, lr_decay):

    with tf.name_scope('optimization'):
        # Compute gradients.
        tvars = tf.trainable_variables()
        grads = tf.gradients(
            loss, tvars,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        # Clip non-embedding grads
        non_embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
            if 'embedding' not in v.op.name]
        embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
            if 'embedding' in v.op.name]

        ne_grads, ne_vars = zip(*non_embedding_grads_and_vars)
        ne_grads, _ = tf.clip_by_global_norm(ne_grads, max_grad_norm)
        non_embedding_grads_and_vars = list(zip(ne_grads, ne_vars))


        grads_and_vars = embedding_grads_and_vars + non_embedding_grads_and_vars

        # Summarize
        _summarize_vars_and_grads(grads_and_vars)

        # Decaying learning rate
        lr = tf.train.exponential_decay(
            lr, global_step, 1, lr_decay, staircase=True)
        tf.summary.scalar('learning_rate', lr)
        opt = tf.train.AdamOptimizer(lr)

        #print ("lalal____________")
        #print (grads_and_vars)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)

        # Apply gradients
        # Non-sync optimizer
        apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = variable_averages.apply(tvars)

        return train_op

def _summarize_vars_and_grads(grads_and_vars):
    tf.logging.info('Trainable variables:')
    tf.logging.info('-' * 60)
    for grad, var in grads_and_vars:
        tf.logging.info(var)

        def tag(name, v=var):
            return v.op.name + '_' + name

        # Variable summary
        mean = tf.reduce_mean(var)
        tf.summary.scalar(tag('mean'), mean)
        with tf.name_scope(tag('stddev')):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(tag('stddev'), stddev)
        tf.summary.scalar(tag('max'), tf.reduce_max(var))
        tf.summary.scalar(tag('min'), tf.reduce_min(var))
        tf.summary.histogram(tag('histogram'), var)

        # Gradient summary
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                grad_values = grad.values
            else:
                grad_values = grad

            tf.summary.histogram(tag('gradient'), grad_values)
            tf.summary.scalar(tag('gradient_norm'), tf.global_norm([grad_values]))
        else:
            tf.logging.info('Var %s has no gradient', var.op.name)

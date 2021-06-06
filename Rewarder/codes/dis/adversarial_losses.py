"""Adversarial losses for text models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange
import tensorflow as tf

from Rewarder.codes.dis.config import FLAGS

def adversarial_loss_bidir(embedded, loss, loss_fn):
    """Adds gradient to embeddings and recomputes classification loss."""
    grads = tf.gradients(
        loss,
        embedded,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    adv_exs = [
        emb + _scale_l2(tf.stop_gradient(g), FLAGS.perturb_norm_length)
        for emb, g in zip(embedded, grads)
    ]
    return loss_fn(adv_exs)

def virtual_adversarial_loss_bidir(logits, embedded, inputs,
    logits_from_embedding_fn, maxlen):
    """Virtual adversarial loss for bidirectional models."""
    logits = tf.stop_gradient(logits)
    f_inputs, r_inputs = inputs
    weights = f_inputs['eos_weights']
    '''
    if FLAGS.single_label:
        indices = tf.stack([tf.range(FLAGS.batch_size), f_inputs['length'] - 1], 1)
        weights = tf.expand_dims(tf.gather_nd(f_inputs['eos_weights'], indices), 1)
    '''
    assert weights is not None

    perturbs = [
        _mask_by_length(tf.random_normal(shape=tf.shape(emb)), f_inputs['length'], maxlen)
        for emb in embedded
    ]

    for _ in xrange(FLAGS.num_power_iteration):
        perturbs = [
            _scale_l2(d, FLAGS.small_constant_for_finite_diff) for d in perturbs
        ]
        d_logits = logits_from_embedding_fn(
            [emb + d for (emb, d) in zip(embedded, perturbs)])
        kl = _kl_divergence_with_logits(logits, d_logits, weights)
        perturbs = tf.gradients(
            kl, perturbs,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        perturbs = [tf.stop_gradient(d) for d in perturbs]

    perturbs = [_scale_l2(d, FLAGS.perturb_norm_length) for d in perturbs]
    vadv_logits = logits_from_embedding_fn(
        [emb + d for (emb, d) in zip(embedded, perturbs)])
    return _kl_divergence_with_logits(logits, vadv_logits, weights)


#--------------
# tool funcitons
def _scale_l2(x, norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit

def _mask_by_length(t, length, maxlen):
    """Mask t, 3-D [batch, time, dim], by length, 1-D [batch,]."""
    # Subtract 1 from length to prevent the perturbation from going on 'eos'
    mask = tf.sequence_mask(length - 1, maxlen=maxlen)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
    # shape(mask) = (batch, num_timesteps, 1)
    return tf.multiply(t, mask)

def _kl_divergence_with_logits(q_logits, p_logits, weights):
    """Returns weighted KL divergence between distributions q and p.
    Args:
        q_logits: logits for 1st argument of KL divergence shape
              [batch_size, num_timesteps, num_classes] if num_classes > 2, and
              [batch_size, num_timesteps] if num_classes == 2.
        p_logits: logits for 2nd argument of KL divergence with same shape q_logits.
        weights: 1-D float tensor with shape [batch_size, num_timesteps].
            Elements should be 1.0 only on end of sequences

    Returns:
        KL: float scalar.
    """
    # For logistic regression
    if FLAGS.num_classes == 2:
        q = tf.nn.sigmoid(q_logits)
        kl = (-tf.nn.sigmoid_cross_entropy_with_logits(logits=q_logits, labels=q) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=q))
        kl = tf.squeeze(kl, 2)

    # For softmax regression
    else:
        q = tf.nn.softmax(q_logits)
        kl = tf.reduce_sum(
            q * (tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits)), -1)

    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)

    kl.get_shape().assert_has_rank(2)
    weights.get_shape().assert_has_rank(2)

    loss = tf.identity(tf.reduce_sum(weights * kl) / num_labels, name='kl')
    return loss










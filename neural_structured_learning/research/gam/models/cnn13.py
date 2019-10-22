# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A convolutional neural network architecture for image classification.

This CNN contains 13 layers and is inspired from the architecture used in
https://github.com/takerum/vat_tf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gam.models.models_base import Model

import numpy as np
import tensorflow as tf


def lrelu(x, a=0.1):
  """Leaky ReLU."""
  if a < 1e-16:
    return tf.nn.relu(x)
  else:
    return tf.maximum(x, a * x)


def conv(x, ksize, stride, f_in, f_out, padding='SAME', use_bias=False,
         seed=None, name='conv'):
  """Convolution layer."""
  shape = [ksize, ksize, f_in, f_out]
  initializer = tf.contrib.layers.variance_scaling_initializer(seed=seed)
  weights = tf.get_variable(name + '_W',
                            shape=shape,
                            dtype='float',
                            initializer=initializer)
  x = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding)

  if use_bias:
    bias = tf.get_variable(name + '_b',
                           shape=[f_out],
                           dtype='float',
                           initializer=tf.zeros_initializer)
    return tf.nn.bias_add(x, bias)
  else:
    return x


def bn(x, dim, bn_stats_decay_factor, is_training, update_batch_stats=True,
       collections=None, name="bn"):
  """Batch norm layer."""
  params_shape = (dim,)
  n = tf.to_float(tf.reduce_prod(tf.shape(x)[:-1]))
  axis = list(range(int(tf.shape(x).get_shape().as_list()[0]) - 1))
  mean = tf.reduce_mean(x, axis)
  var = tf.reduce_mean(tf.pow(x - mean, 2.0), axis)
  avg_mean = tf.get_variable(
      name=name + "_mean",
      shape=params_shape,
      initializer=tf.constant_initializer(0.0),
      collections=collections,
      trainable=False)

  avg_var = tf.get_variable(
      name=name + "_var",
      shape=params_shape,
      initializer=tf.constant_initializer(1.0),
      collections=collections,
      trainable=False)

  gamma = tf.get_variable(
      name=name + "_gamma",
      shape=params_shape,
      initializer=tf.constant_initializer(1.0),
      collections=collections)

  beta = tf.get_variable(
      name=name + "_beta",
      shape=params_shape,
      initializer=tf.constant_initializer(0.0),
      collections=collections)

  def bn_train():
    """Batch norm implementation for training mode."""
    avg_mean_assign_op_1 = tf.identity(avg_mean)
    avg_var_assign_op_1 = tf.identity(avg_var)
    avg_mean_assign_op_2 = tf.assign(
        avg_mean,
        bn_stats_decay_factor * avg_mean +
        (1 - bn_stats_decay_factor) * mean)
    avg_var_assign_op_2 = tf.assign(
        avg_var,
        bn_stats_decay_factor * avg_var + (n / (n - 1)) *
        (1 - bn_stats_decay_factor) * var)
    avg_mean_assign_op = tf.cond(
        update_batch_stats,
        lambda: avg_mean_assign_op_2,
        lambda: avg_mean_assign_op_1)
    avg_var_assign_op = tf.cond(
        update_batch_stats,
        lambda: avg_var_assign_op_2,
        lambda: avg_var_assign_op_1)

    with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
        z = (x - mean) / tf.sqrt(1e-6 + var)
        return z

  def bn_test():
    """Batch norm implementation for test mode."""
    z = (x - avg_mean) / tf.sqrt(1e-6 + avg_var)
    return z

  z = tf.cond(is_training, bn_train, bn_test)

  return gamma * z + beta


def max_pool(x, ksize=2, stride=2):
  """Max pooling layer."""
  return tf.nn.max_pool(
      x,
      ksize=[1, ksize, ksize, 1],
      strides=[1, stride, stride, 1],
      padding='SAME')


def fc(x, dim_in, dim_out, seed=None, name='fc'):
  """Fully connected layer."""
  num_units_in = dim_in
  num_units_out = dim_out
  weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      seed=seed)

  weights = tf.get_variable(
      name + '_W',
      shape=[num_units_in, num_units_out],
      initializer=weights_initializer)
  biases = tf.get_variable(
      name + '_b',
      shape=[num_units_out],
      initializer=tf.constant_initializer(0.0))
  x = tf.nn.xw_plus_b(x, weights, biases)
  return x


class ImageCNN13(Model):
  """Convolutional Neural Network with 13 layers for image classification.

  It assumes the inputs are images of shape width x height x channels.
  The precise architecture is the same as the one used in several
  semi-supervised learning publications. This implementation is based on the
  code released with Virtual Adversarial Training:
  https://github.com/takerum/vat_tf/blob/master/cnn.py.


  Note that this CNN works both for the agreement and classification models.
  In the agreement case, the provided inputs will be a tuple
  (inputs_src, inputs_target), which are aggregated into one input after the
  convolution layers, right before the fully connected network that makes the
  final prediction.

  Attributes:
    output_dim: Integer representing the number of classes.
    aggregation: String representing an aggregation operation, that is applied
      on the two inputs of the agreement model, after they are encoded through
      the convolution layers. See superclass attributes for details.
    hidden_aggregation: A list of integers representing the number of units of
      each hidden layer of the aggregation network.
    activation: An activation function to be applied to the outputs of each
      fully connected layer of the aggregation network.
    is_binary_classification: Boolean specifying if this is model for
      binary classification. If so, it uses a different loss function and
      returns predictions with a single dimension, batch size.
    lrelu_a=0.1,
    keep_prob_hidden=0.5,
    top_bn=False,
    bn_stats_decay_factor=0.99,
    name: String representing the model name.
  """

  def __init__(self,
               output_dim,
               aggregation=None,
               hidden_aggregation=(),
               activation=tf.nn.leaky_relu,
               is_binary_classification=False,
               lrelu_a=0.1,
               keep_prob_hidden=0.5,
               top_bn=False,
               bn_stats_decay_factor=0.99,
               seed=None,
               name='cnn13'):
    super(ImageCNN13, self).__init__(
        aggregation=aggregation,
        hidden_aggregation=hidden_aggregation,
        activation=activation)
    self.output_dim = output_dim
    self.is_binary_classification = is_binary_classification
    self.lrelu_a = lrelu_a
    self.keep_prob_hidden = keep_prob_hidden
    self.top_bn = top_bn
    self.bn_stats_decay_factor = bn_stats_decay_factor
    self.name = name

    # Generate some random numbers used for initializing the weights.
    rng = np.random.RandomState(seed)
    self.random_numbers = rng.randint(0, 123456, size=(11,))


  def _get_encoding(self, inputs, is_train, update_batch_stats):
    """Creates all hidden layers of the model, before the prediction layer.

    Args:
      inputs: A tensor containing the model inputs. The first dimension is the
        batch size.
      is_train: A boolean specifying whether this is train or test.
      update_batch_stats: A boolean specifying whether to update the batch norm
        statistics.
      rng: A RandomState object.

    Returns:
      A tuple containing the encoded representation of the inputs and a
      dictionary of regularization parameters.
    """
    reg_params = {}
    h = inputs
    no_dropout = tf.cast(tf.logical_not(is_train), tf.float32)

    h = conv(
            h, ksize=3, stride=1, f_in=3, f_out=128, seed=self.random_numbers[0],
            name='c1')
    h = lrelu(
            bn(
                h, 128, bn_stats_decay_factor=self.bn_stats_decay_factor,
                is_training=is_train,
                update_batch_stats=update_batch_stats, name='b1'),
            self.lrelu_a)
    h = conv(
            h, ksize=3, stride=1, f_in=128, f_out=128,
            seed=self.random_numbers[1], name='c2')
    h = lrelu(
            bn(
                h, 128, bn_stats_decay_factor=self.bn_stats_decay_factor,
                is_training=is_train, update_batch_stats=update_batch_stats,
                name='b2'),
            self.lrelu_a)
    h = conv(
            h, ksize=3, stride=1, f_in=128, f_out=128,
            seed=self.random_numbers[2], name='c3')
    h = lrelu(
            bn(
                h, 128, bn_stats_decay_factor=self.bn_stats_decay_factor,
                is_training=is_train, update_batch_stats=update_batch_stats,
                name='b3'),
            self.lrelu_a)

    h = max_pool(h, ksize=2, stride=2)
    keep_prob_hidden = tf.maximum(no_dropout, self.keep_prob_hidden)
    h = tf.nn.dropout(
        h, keep_prob=keep_prob_hidden, seed=self.random_numbers[3])
    h = conv(
            h, ksize=3, stride=1, f_in=128, f_out=256,
            seed=self.random_numbers[4], name='c4')
    h = lrelu(
            bn(
                h, 256, bn_stats_decay_factor=self.bn_stats_decay_factor,
                is_training=is_train, update_batch_stats=update_batch_stats,
                name='b4'),
            self.lrelu_a)
    h = conv(
            h, ksize=3, stride=1, f_in=256, f_out=256,
            seed=self.random_numbers[5], name='c5')
    h = lrelu(
            bn(
                h, 256, bn_stats_decay_factor=self.bn_stats_decay_factor,
                is_training=is_train, update_batch_stats=update_batch_stats,
                name='b5'),
            self.lrelu_a)
    h = conv(
            h, ksize=3, stride=1, f_in=256, f_out=256,
            seed=self.random_numbers[6], name='c6')
    h = lrelu(
            bn(
                h, 256, bn_stats_decay_factor=self.bn_stats_decay_factor,
                is_training=is_train, update_batch_stats=update_batch_stats,
                name='b6'),
            self.lrelu_a)
    h = max_pool(h, ksize=2, stride=2)
    h = tf.nn.dropout(
            h, keep_prob=keep_prob_hidden, seed=self.random_numbers[7])
    h = conv(
            h, ksize=3, stride=1, f_in=256, f_out=512,
            seed=self.random_numbers[8], padding="VALID", name='c7')
    h = lrelu(
            bn(h, 512, bn_stats_decay_factor=self.bn_stats_decay_factor,
                 is_training=is_train,
                 update_batch_stats=update_batch_stats, name='b7'),
              self.lrelu_a)
    h = conv(
        h, ksize=1, stride=1, f_in=512, f_out=256, seed=self.random_numbers[9],
        name='c8')
    h = lrelu(
        bn(
            h, 256, bn_stats_decay_factor=self.bn_stats_decay_factor,
            is_training=is_train, update_batch_stats=update_batch_stats,
            name='b8'),
        self.lrelu_a)
    h = conv(
        h, ksize=1, stride=1, f_in=256, f_out=128, seed=self.random_numbers[10],
        name='c9')
    h = lrelu(
        bn(
            h, 128, bn_stats_decay_factor=self.bn_stats_decay_factor,
            is_training=is_train, update_batch_stats=update_batch_stats,
            name='b9'),
         self.lrelu_a)

    return h, reg_params

  def get_encoding_and_params(self, inputs, update_batch_stats=True, **kwargs):
    """Creates the model hidden representations and prediction ops.

    For this model, the hidden representation is the last layer
    before the logit computation. The predictions are unnormalized logits.

    Args:
      inputs: A tensor containing the model inputs. The first dimension is the
        batch size.
      update_batch_stats: Boolean specifying whether to update the batch norm
        statistics.
      **kwargs: Other keyword arguments.

    Returns:
      encoding: A tensor containing an encoded batch of samples. The first
        dimension corresponds to the batch size.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    is_train = kwargs['is_train']

    # Build layers.
    with tf.variable_scope(self.name):
      if isinstance(inputs, (list, tuple)):
        # If we have multiple inputs (e.g., in the case of the agreement model),
        # split into left and right inputs, compute the hidden representation of
        # each branch, then aggregate.
        left = inputs[0]
        right = inputs[1]
        with tf.variable_scope('encoding'):
          hidden1, reg_params = self._get_encoding(
              left, is_train, update_batch_stats)
        with tf.variable_scope('encoding', reuse=True):
          hidden2, _ = self._get_encoding(
              right, is_train, update_batch_stats)
        encoding = self._aggregate((hidden1, hidden2))
      else:
        with tf.variable_scope('encoding'):
          encoding, reg_params = self._get_encoding(
              inputs, is_train, update_batch_stats)

      # Store model variables for easy access.
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES,
          scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

    return encoding, all_vars, reg_params

  def get_predictions_and_params(self, encoding, is_train, **kwargs):
    """Creates the model prediction op.

    For this model, the hidden representation is the last layer
    before the logit computation. The predictions are unnormalized logits.

    Args:
      encoding: A tensor containing the model inputs. The first dimension is the
        batch size.
      is_train: A placeholder representing a boolean value that specifies if
        this model will be used for training or for test.
      **kwargs: Other keyword arguments.

    Returns:
      predictions: A tensor of logits. For multiclass classification its
        shape is (num_samples, num_classes), where the second dimension contains
        a logit per class. For binary classification, its shape is
        (num_samples,), where each element is the probability of class 1 for
        that sample.
      all_vars: A dictionary mapping from variable name to TensorFlow op
        containing all variables used in this model.
      reg_params: A dictionary mapping from a variable name to a Tensor of
        parameters which will be used for regularization.
    """
    update_batch_stats = kwargs['update_batch_stats']

    # Build layers.
    with tf.variable_scope(self.name + '/prediction'):
      # We store all variables on which we apply weight decay in a dictionary.
      reg_params = {}

      # Global average pooling.
      encoding = tf.reduce_mean(encoding, reduction_indices=[1, 2])

      # Logits Layer.
      logits = fc(encoding, 128, self.output_dim, name='fc')
      if self.top_bn:
        logits = bn(
            logits,
            self.output_dim,
            bn_stats_decay_factor=self.bn_stats_decay_factor,
            is_training=is_train,
            update_batch_stats=update_batch_stats,
            name='predictions')

      if self.is_binary_classification:
        logits = logits[:, 0]

      # Store model variables for easy access.
      variables = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES,
          scope=tf.get_default_graph().get_name_scope())
      all_vars = {var.name: var for var in variables}

    return logits, all_vars, reg_params

  def get_loss(self,
               predictions,
               targets,
               name_scope='loss',
               reg_params=None,
               **kwargs):
    """Returns a loss between the provided targets and predictions.

    For binary classification, this loss is sigmoid cross entropy. For
    multi-class classification, it is softmax cross entropy.
    A weight decay loss is also added to the parameters passed in reg_params.

    Arguments:
      predictions: A tensor of predictions. For multiclass classification its
        shape is (num_samples, num_classes), where the second dimension contains
        a logit per class. For binary classification, its shape is
        (num_samples,), where each element is the probability of class 1 for
        that sample.
      targets: A tensor of targets of shape (num_samples,), where each row
        contains the label index of the corresponding sample.
      name_scope: A string containing the name scope used in TensorFlow.
      reg_params: A dictonary of parameters, mapping from name to parameter, for
        the variables to be included in the weight decay loss. If None, no
        weight decay is applied.
      **kwargs: Keyword arguments, potentially containing the weight of the
        regularization term, passed under the name `weight_decay`. If this is
        not provided, it defaults to 0.004.

    Returns:
      loss: The cummulated loss value.
    """
    reg_params = reg_params if reg_params is not None else {}
    weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else None

    with tf.name_scope(name_scope):
      # Cross entropy error.
      if self.is_binary_classification:
        loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=targets, logits=predictions))
      else:
        loss = tf.losses.softmax_cross_entropy(targets, predictions)

      # Weight decay loss.
      if weight_decay is not None:
        for var in reg_params.values():
          loss += weight_decay * tf.nn.l2_loss(var)
    return loss

  def normalize_predictions(self, predictions):
    """Converts predictions to probabilities.

    Arguments:
      predictions: A tensor of logits. For multiclass classification its shape
        is (num_samples, num_classes), where the second dimension contains a
        logit per class. For binary classification, its shape is (num_samples,),
        where each element is the probability of class 1 for that sample.

    Returns:
      A tensor of the same shape as predictions, with values between [0, 1]
    representing probabilities.
    """
    if self.is_binary_classification:
      return tf.nn.sigmoid(predictions)
    return tf.nn.softmax(predictions, axis=-1)

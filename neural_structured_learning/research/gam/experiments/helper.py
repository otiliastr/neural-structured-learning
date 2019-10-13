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
"""Helper functions for GAMs."""
from gam.models.cnn import ImageCNNAgreement
from gam.models.cnn13 import ImageCNN13
from gam.models.mlp import MLP
from gam.models.wide_resnet import WideResnet

import tensorflow as tf


def parse_layers_string(layers_string):
  """Convert a layer size string (e.g., `128_64_32`) to a list of integers."""
  if not layers_string:
    return ()
  num_hidden = layers_string.split('_')
  num_hidden = [int(num) for num in num_hidden]
  return num_hidden


def get_model_cls(model_name, data, dataset_name, hidden=None, **kwargs):
  """Picks the models depending on the provided configuration flags."""
  # Create model classification.
  if model_name == 'mlp':
    hidden = parse_layers_string(hidden) if hidden is not None else ()
    return MLP(
        output_dim=data.num_classes,
        hidden_sizes=hidden,
        activation=tf.nn.leaky_relu,
        name='mlp_cls')
  elif model_name == 'cnn':
    if dataset_name in ('mnist', 'fashion_mnist'):
      channels = 1
    elif dataset_name in ('cifar10', 'cifar100', 'svhn_cropped', 'svhn'):
      channels = 3
    else:
      raise ValueError('Dataset name `%s` unsupported.' % dataset_name)
    return ImageCNNAgreement(
        output_dim=data.num_classes,
        channels=channels,
        activation=tf.nn.leaky_relu,
        name='cnn_cls')
  elif model_name == 'wide_resnet':
    return WideResnet(
        num_classes=data.num_classes,
        lrelu_leakiness=0.1,
        horizontal_flip=dataset_name in ('cifar10',),
        random_translation=True,
        gaussian_noise=dataset_name not in ('svhn', 'svhn_cropped'),
        width=2,
        num_residual_units=4,
        name='wide_resnet_cls')
  elif model_name == 'cnn13':
    return ImageCNN13(
        output_dim=data.num_classes,
        seed=kwargs['seed'],
        name='cnn13_cls')
  else:
    raise NotImplementedError()


def get_model_agr(
      model_name,
      dataset_name,
      hidden_aggreg=None,
      aggregation_agr_inputs='dist',
      hidden=None,
      **kwargs):
  """Create agreement model."""
  hidden = parse_layers_string(hidden) if hidden is not None else ()
  hidden_aggreg = (
      parse_layers_string(hidden_aggreg) if hidden_aggreg is not None else ())
  if model_name == 'mlp':
    return MLP(
        output_dim=1,
        hidden_sizes=hidden,
        activation=tf.nn.leaky_relu,
        aggregation=aggregation_agr_inputs,
        hidden_aggregation=hidden_aggreg,
        is_binary_classification=True,
        name='mlp_agr')
  elif model_name == 'cnn':
    if dataset_name in ('mnist', 'fashion_mnist'):
      channels = 1
    elif dataset_name in ('cifar10', 'cifar100', 'svhn_cropped', 'svhn'):
      channels = 3
    else:
      raise ValueError('Dataset name `%s` unsupported.' % dataset_name)
    return ImageCNNAgreement(
        output_dim=1,
        channels=channels,
        activation=tf.nn.leaky_relu,
        aggregation=aggregation_agr_inputs,
        hidden_aggregation=hidden_aggreg,
        is_binary_classification=True,
        name='cnn_agr')
  elif model_name == 'wide_resnet':
    return WideResnet(
        num_classes=1,
        lrelu_leakiness=0.1,
        horizontal_flip=dataset_name in ('cifar10',),
        random_translation=True,
        gaussian_noise=dataset_name not in ('svhn', 'svhn_cropped'),
        width=2,
        num_residual_units=4,
        name='wide_resnet_cls',
        is_binary_classification=True,
        aggregation=aggregation_agr_inputs,
        activation=tf.nn.leaky_relu,
        hidden_aggregation=hidden_aggreg)
  elif model_name == 'cnn13':
    return ImageCNN13(
        output_dim=1,
        seed=kwargs['seed'],
        is_binary_classification=True,
        aggregation=aggregation_agr_inputs,
        activation=tf.nn.leaky_relu,
        hidden_aggregation=hidden_aggreg,
        name='cnn13_agr')
  else:
    raise NotImplementedError()

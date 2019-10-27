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
"""Functions for data preprocessing.

This file contains functions for preprocessing data, such as splitting into
train, validation and unlabeled samples.
"""

from __future__ import absolute_import

import collections
import numpy as np


def convert_image(image):
  """Convert an image from pixels in {0, ..., 255} to floats in [-1, 1]."""
  image = image * (1. / 255) - 0.5
  image *= 2.
  return image


def split_train_val(indices, ratio_val, rng, max_num_val=None):
  """Split the train sample indices into train and validation.

  Args:
    indices: A numpy array containing the indices of the training samples.
    ratio_val: A float number between (0, 1) representing the ratio of samples
      to use for validation.
    rng: A random number generator.
    max_num_val: An integer representing the maximum number of samples to
      include in the validation set.

  Returns:
    Two numpy arrays containing the subset of indices used for training, and
    validation, respectively.
  """
  num_samples = indices.shape[0]
  num_val = int(ratio_val * num_samples)
  if max_num_val and num_val > max_num_val:
    num_val = max_num_val
  ind = np.arange(0, num_samples)
  rng.shuffle(ind)
  ind_val = ind[:num_val]
  ind_train = ind[num_val:]
  return ind_train, ind_val


def split_train_val_unlabeled(train_inputs,
                              train_labels,
                              target_num_train_per_class,
                              target_num_val,
                              seed=None):
  """Splits the training data into train, validation and unlabeled samples.

  Arguments:
      train_inputs: A numpy array containing the training inputs, where the
        first dimension represents the samples.
      train_labels: A numpy array containing the training labels, where the
        first dimension represents the samples.
      target_num_train_per_class: An integer representing the number of samples
        from each class to keep for training.
      target_num_val: An integer representing the number of samples to keep for
        validation (in total). We do not make the selection per class.
      seed: Integer representing the seed for the random number generator that
        splits the data.

  Returns:
    train_inputs: A numpy array in the same format as `train_inputs` containing
      the new train inputs.
    train_labels: A numpy array in the same format as `train_labels` containing
      the new train labels.
    val_inputs: A numpy array in the same format as `train_inputs` containing
      the validation inputs.
    val_labels: A numpy array in the same format as `train_labels` containing
      the validation labels.
    unlabeled_inputs: A numpy array in the same format as `train_inputs`
      containing the inputs for the unlabeled nodes.
    unlabeled_labels: A numpy array in the same format as `train_labels`
      containing the labels of the samples that will be considered unlabeled.
  """
  num_train = train_inputs.shape[0]
  num_val = target_num_val
  num_classes = max(train_labels) + 1

  assert target_num_val < num_train, 'Too many validation samples required.'

  # Split the train samples into train and validation.
  ind = np.arange(0, num_train)
  rng = np.random.RandomState(seed)
  rng.shuffle(ind)
  ind_val = ind[:num_val]
  ind_train = ind[num_val:]
  val_inputs = train_inputs[ind_val]
  val_labels = train_labels[ind_val]
  train_inputs = train_inputs[ind_train]
  train_labels = train_labels[ind_train]

  # Out of the remaining training samples, select a fixed number of samples
  # from each class.
  ind_train = []
  for i in range(num_classes):
    ind_class = np.where(train_labels == i)[0]
    assert len(ind_class) >= target_num_train_per_class, \
           ('Not enough labels for class %d to select %d labels per class. '
            'Please select a target_num_train_per_class flag lower than %d.' %
            (i, target_num_train_per_class, len(ind_class)))
    rng.shuffle(ind_class)
    selected_ind = ind_class[:target_num_train_per_class]
    ind_train.extend(selected_ind)

  # Having selected the train indices, save the remaining unlabeled nodes in a
  # different group.
  set_ind_train = set(ind_train)
  num_train = num_train - num_val
  ind_unlabeled = [i for i in range(num_train) if i not in set_ind_train]
  ind_train = np.asarray(ind_train)
  ind_unlabeled = np.asarray(ind_unlabeled)
  unlabeled_inputs = train_inputs[ind_unlabeled]
  unlabeled_labels = train_labels[ind_unlabeled]
  train_inputs = train_inputs[ind_train]
  train_labels = train_labels[ind_train]

  return (train_inputs, train_labels, val_inputs, val_labels, unlabeled_inputs,
          unlabeled_labels)


def random_splits(data):
  """Shuffle the data to pick random train/val/test/unlabeled splits.

  When doing the shuffling, we keep the original number of samples per class
  in each of the train/val/test/unlabeled sets, but these are chosen at random
  from the entire data pool. For example, if the original dataset had
  20 samples per label, the output dataset will also have 20 per label,
  but the specific samples are chosen at random from the union of train,
  val and test sets.

  Arguments:
      data: A Dataset object.
  """
  def _shuffle(current_indices):
    """Shuffle the dataset, replacing the current indices with others with
    the same label.

    :param current_indices: Current sample indices that are part of this
        split.
    :return:
        A new set of sample indices, that have the same number of samples
         per label as `current_indices`.
    """
    new_indices = []

    # Compute the number of samples per class.
    labels = data.get_labels(current_indices)
    num_per_label = collections.Counter(labels)

    # Select the corresponding number of samples for each label.
    for label in range(data.num_classes):
      if label not in num_per_label:
        continue
      num_to_select = num_per_label[label]
      end_index = start_idx_per_label[label] + num_to_select
      selected_indices = indices_per_label[label][
                         start_idx_per_label[label]:end_index]
      new_indices.extend(selected_indices)
      start_idx_per_label[label] = end_index

    return new_indices

  # Take all samples and separate per label.
  indices = np.arange(data.num_samples)
  labels = data.get_labels(indices)
  indices_per_label = [[] for _ in range(data.num_classes)]
  for index, label in zip(indices, labels):
    indices_per_label[label].append(index)

  # Shuffle the indices per label.
  for label in range(data.num_classes):
    np.random.shuffle(indices_per_label[label])

  # Compute how many samples per label we should have in each of the
  # train/val/test sets. Then, for each label, split the randomly shuffled
  # indices_per_label[label] between train/val/test according to these counts.
  start_idx_per_label = [0] * data.num_classes

  # Select the training set.
  new_ind = _shuffle(data.get_indices_train())
  data.set_indices_train(new_ind)

  # Select the validation set.
  new_ind = _shuffle(data.get_indices_val())
  data.set_indices_val(new_ind)

  # Select the test set.
  new_ind = _shuffle(data.get_indices_test())
  data.set_indices_test(new_ind)

  # The remaining samples are unlabeled.
  new_ind = []
  for label in range(data.num_classes):
    selected_indices = indices_per_label[label][start_idx_per_label[label]:]
    new_ind.extend(selected_indices)
  data.set_indices_unlabeled(new_ind)
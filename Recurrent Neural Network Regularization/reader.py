# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
    '''Generate a word_to_id dictionary.

    Check out test.ipynb for more details.
    '''
    data = _read_words(filename)
    word_to_id = {}
    word_to_id['UNK'] = 0
    word_count_sorted = sorted(collections.Counter(data).items(), key=lambda item: item[1])
    for item in word_count_sorted:
        if item[1] > 3: # if word frequency > 3
            word_to_id[item[0]] = len(word_to_id) # index word by dictionary length
        else:
            word_to_id['UNK'] += 1
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id.get(word, 0) for word in data]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "wiki.train.txt")
  valid_path = os.path.join(data_path, "wiki.valid.txt")
  test_path = os.path.join(data_path, "wiki.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y


def generate_tokens(data_path, token_class):
    # create word-id dictionary
    word_to_id = _build_vocab(os.path.join(data_path, "wiki.train.txt"))

    # generate token ids
    tens = [i for i in range(10, 100, 10)]
    hundreds = [i for i in range(100, 1000, 100)]
    rounds = set([word_to_id.get(str(i), 0) for i in tens + hundreds])
    days = set([word_to_id.get(str(i), 0) for i in range(1, 32, 1)])
    years = set([word_to_id.get(str(i), 0) for i in range(1000, 2021, 1)])

    if token_class == 'rounds':
        return rounds
    elif token_class == 'days':
        return days
    else:
        return years


def generator(data, batch_size, num_steps, tokens):
    '''Generate token positions iteratively.

    The shape of data is [batch_size, batch_len], obtained in the same way as
    the author did in function 'ptb_producer' (line 88).

    During training/validaiton/test, loss and perplexity is calculated per
    slice of data in shape of [batch_size, num_steps]
    shifting on data from left to right.

    For each slice of data, token position [j, k, data[j][i + k]] will be used to
    extract the corresponding probability from the 'prob' tensor derived from logits,
    whose shape is [batch_size, num_steps, vocab_size] (line 137 in ptb_word_lm.py).

    Note that data[j][i + k] is in range [0, vocab_size] and represent
    a token id.
    '''
    data_len = len(data)
    batch_len = data_len // batch_size
    data = np.reshape(data[0: batch_size * batch_len], [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps

    i = 1
    while True:
        positions = []

        for j in range(batch_size):
            for k in range(num_steps):
                if data[j][i + k] in tokens:
                    positions.append((j, k, data[j][i + k]))

        if i == batch_len - num_steps - 1:
            i = 1
        else:
            i += 1

        yield positions

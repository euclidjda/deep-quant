# ==============================================================================
# Copyright 2016 Euclidean Technologies Management LLC All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from deep_nn_model import DeepNNModel
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

class NaiveModel(DeepNNModel):
  """
  """
  def __init__(self, config):
      """
      Initialize the model
      Args:
        config
      """

      self._num_unrollings = num_unrollings = config.num_unrollings
      self._num_inputs = num_inputs = config.num_features
      num_outputs = num_inputs
      
      total_input_size = num_unrollings * num_inputs

      # input/target normalization params
      self._center = tf.get_variable('center',shape=[num_inputs],trainable=False)
      self._scale  = tf.get_variable('scale',shape=[num_inputs],trainable=False)
      
      batch_size = self._batch_size = tf.placeholder(tf.int32, shape=[])
      self._keep_prob = tf.placeholder(tf.float32, shape=[])
      self._phase = tf.placeholder(tf.bool, name='phase')
      
      self._inputs = list()
      self._targets = list()

      for _ in range(num_unrollings):
        self._inputs.append( tf.placeholder(tf.float32,
                                              shape=[None,num_inputs]) )
        self._targets.append( tf.placeholder(tf.float32,
                                              shape=[None,num_outputs]) )

      outputs = self._inputs[-1]        
      targets = self._targets[-1]
      
      # center and scale
      if config.data_scaler is not None:
        targets = tf.divide(targets - self._center, self._scale)
        outputs = tf.divide(outputs - self._center, self._scale)

      self._o = outputs
      self._t = targets
      
      self._mse = tf.losses.mean_squared_error(targets, outputs)

      if config.data_scaler is not None:
        self._predictions = tf.multiply(outputs,self._scale) + self._center
      else:
        self._predictions = outputs

      self._lr = tf.Variable(0.0, trainable=False)
      self._train_op = tf.identity(self._lr)

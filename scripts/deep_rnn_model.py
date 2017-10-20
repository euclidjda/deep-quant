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

class DeepRnnModel(DeepNNModel):
  """
  A Deep Rnn Model that supports a binary (two class) output with an
  arbitrary number of fixed width hidden layers.
  """
  def __init__(self, config):
      """
      Initialize the model
      Args:
        config
      """

      self._num_unrollings = num_unrollings = config.num_unrollings
      self._num_inputs = num_inputs = config.num_inputs
      self._num_outputs = num_outputs = config.num_outputs
      num_hidden = config.num_hidden

      # total_input_size = num_unrollings * num_inputs
      # input/target normalization params
      self._center = tf.get_variable('center',shape=[num_inputs],trainable=False)
      self._scale  = tf.get_variable('scale',shape=[num_inputs],trainable=False)
      
      batch_size = self._batch_size = tf.placeholder(tf.int32, shape=[])
      self._keep_prob = tf.placeholder(tf.float32, shape=[])
      self._phase = tf.placeholder(tf.bool, name='phase')

      self._inputs = list()
      self._targets = list()
 
      for _ in range(num_unrollings):
        inp = tf.placeholder(tf.float32, shape=[None,num_inputs])
        tar =  tf.placeholder(tf.float32, shape=[None,num_outputs])
        self._inputs.append( inp )
        self._targets.append( tar )

      self._scaled_inputs = [None]*num_unrollings
      self._scaled_targets = [None]*num_unrollings
        
      for i in range(num_unrollings):
        if config.data_scaler is not None:
          self._scaled_inputs[i] = self._center_and_scale( self._inputs[i] )
          self._scaled_targets[i] = self._center_and_scale( self._targets[i] )
        else:
          self._scaled_inputs[i] = self._inputs[i]
          self._scaled_targets[i] = self._targets[i]
            
      hkp = self._keep_prob if config.hidden_dropout is True else 1.0
      ikp = self._keep_prob if config.input_dropout is True else 1.0

      def rnn_cell():
        cell = tf.contrib.rnn.GRUCell(num_hidden)
        return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=hkp,input_keep_prob=ikp)

      stacked_rnn = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(config.num_layers)])
      
      outputs, state = tf.contrib.rnn.static_rnn(stacked_rnn, self._scaled_inputs, dtype=tf.float32)

      self._w = softmax_w = tf.get_variable("softmax_w", [num_hidden, num_outputs])
      softmax_b = tf.get_variable("softmax_b", [num_outputs])

      self._outputs = list()
      for i in range(num_unrollings):
        self._outputs.append( tf.nn.xw_plus_b( outputs[i], softmax_w, softmax_b ) )

      outputs = tf.concat(self._outputs, 0)
      targets = tf.concat(self._scaled_targets, 0)

      self._o = outputs
      self._t = targets
            
      self._mse_all_steps = tf.losses.mean_squared_error(targets, outputs)
      self._mse = tf.losses.mean_squared_error(self._scaled_targets[-1], self._outputs[-1])
      
      if config.data_scaler is not None:
        self._predictions = self._reverse_center_and_scale( self._outputs[-1] )
      else:
        self._predictions = self._outputs[-1]
 
      # here is the learning part of the graph
      loss = config.rnn_lambda*self._mse + (1.0-config.rnn_lambda)*self._mse_all_steps
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss ,tvars)

      if (config.max_grad_norm > 0):
        grads, self._grad_norm = tf.clip_by_global_norm(grads,config.max_grad_norm)
      else:
        self._grad_norm = tf.constant(0.0)

      self._lr = tf.Variable(0.0, trainable=False)
      optimizer = None
      args = config.optimizer_params
      if hasattr(tf.train,config.optimizer):
        optimizer = getattr(tf.train, config.optimizer)(learning_rate=self._lr,**args)
      else:
        raise RuntimeError("Unknown optimizer = %s"%config.optimizer)
 
      self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    

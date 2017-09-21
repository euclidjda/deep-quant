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

class DeepMlpModel(DeepNNModel):
  """
  A Deep MLP Model that supports a mult-class output with an
  arbitrary number of fixed width hidden layers.
  """
  def __init__(self, config):
      """
      Initialize the model
      Args:
        config
      """
      num_outputs = 2
      self._num_unrollings = num_unrollings = config.num_unrollings
      self._num_inputs = num_inputs = config.num_inputs

      total_input_size = num_unrollings * num_inputs

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
        
      inputs = tf.concat( self._inputs, 1 )

      if config.input_dropout is True: inputs = self._input_dropout(inputs)

      num_prev = total_input_size
 
      outputs = inputs

      for i in range(config.num_layers):
        outputs = self._batch_relu_layer(outputs, config.num_hidden, self._phase, "layer_%d"%i)
        if config.hidden_dropout is True:
          outputs = tf.nn.dropout(outputs, self._keep_prob)
        num_prev = config.num_hidden

      if config.skip_connections is True:
        num_prev = num_inputs+num_prev
        skip_inputs = tf.slice(inputs, [0, 0], [batch_size, num_inputs] )
        outputs  = tf.concat( [ skip_inputs, outputs], 1)

      # final regression layer  
      regress_b = tf.get_variable("softmax_b", [num_outputs])
      regress_w = tf.get_variable("softmax_w", [num_prev, num_outputs])
      self._predictions = tf.nn.xw_plus_b(outputs, regress_w, regress_b)

      # We are just predicting the next (last in targets) time step so we index by -1
      # in self._targets
      self._mse = tf.losses.mean_squared_error(self._targets[-1], self._predictions)
      
      # from here down is the learning part of the graph
      tvars = tf.trainable_variables()
      grads = tf.gradients(self._mse,tvars)

      if (config.max_grad_norm > 0):
        grads, _ = tf.clip_by_global_norm(grads,config.max_grad_norm)

      self._lr = tf.Variable(0.0, trainable=False)
      optimizer = None
      args = self._get_optimizer_args(config.optimizer_params)
      if hasattr(tf.train,config.optimizer):
        optimizer = getattr(tf.train, config.optimizer)(**args)
      else:
        raise RuntimeError("Unknown optimizer = %s"%config.optimizer)

      self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def _get_optimizer_args(self,optimizer_params):
      # optimizer_params is a string of the form "param1=value1,param2=value2,..."
      # this method maps it to dictionary { param1 : value1, param2 : value2, ...}
      args_list = [p.split('=') for p in optimizer_params.split(',')]
      args = dict()
      for p in args_list:
        args[p[0]] = float(p[1])
      return args
      
  def _input_dropout(self,inputs):
    # This implementation of dropout dropouts an entire feature along the time dim
    random_tensor = self._keep_prob
    random_tensor += random_ops.random_uniform([self._batch_size,self._num_inputs],
                                               dtype=inputs.dtype)
    random_tensor = tf.tile(random_tensor,[1,self._num_unrollings])
    binary_tensor = math_ops.floor(random_tensor)

    ret = math_ops.div(inputs, self._keep_prob) * binary_tensor
    ret.set_shape(inputs.get_shape())
    return ret

  def _batch_relu_layer(self, x, size, phase, scope):
    with tf.variable_scope(scope):
      h1 = tf.contrib.layers.fully_connected(x, size,
                                              activation_fn=None,
                                              scope='dense')
      h2 = tf.contrib.layers.batch_norm(h1, 
                                        center=True, scale=True,
                                        is_training=phase,
                                        scope='bn')
      return tf.nn.relu(h2, 'relu')

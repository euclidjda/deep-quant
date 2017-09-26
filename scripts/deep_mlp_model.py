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

      self._num_unrollings = num_unrollings = config.num_unrollings
      self._num_inputs = num_inputs = config.num_inputs
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
        
      inputs = tf.concat( self._inputs, 1 )
      targets = self._targets[-1]
      
      # center and scale
      if config.data_scaler is not None:
        inputs = tf.divide(inputs - tf.tile(self._center,[num_unrollings]),
                             tf.tile(self._scale,[num_unrollings]))
        targets = tf.divide(targets - self._center, self._scale)

      self._t = targets
      
      if config.input_dropout is True: inputs = self._input_dropout(inputs)

      num_prev = total_input_size
 
      outputs = inputs

      for i in range(config.num_layers):
        outputs = self._batch_relu_layer(outputs, config.num_hidden, "layer_%d"%i)
        if config.hidden_dropout is True:
          outputs = tf.nn.dropout(outputs, self._keep_prob)
        num_prev = config.num_hidden

      if config.skip_connections is True:
        num_prev = num_inputs+num_prev
        skip_inputs = tf.slice(inputs, [0, 0], [batch_size, num_inputs] )
        outputs  = tf.concat( [ skip_inputs, outputs], 1)

      # final regression layer  
      linear_b = tf.get_variable("linear_b", [num_outputs])
      linear_w = tf.get_variable("linear_w", [num_prev, num_outputs])
      outputs = tf.nn.xw_plus_b(outputs, linear_w, linear_b)
      
      self._mse = tf.losses.mean_squared_error(targets, outputs)

      if config.data_scaler is not None:
        self._predictions = tf.multiply(outputs,self._scale) + self._center
      else:
        self._predictions = outputs
      
      # from here down is the learning part of the graph
      tvars = tf.trainable_variables()
      grads = tf.gradients(self._mse,tvars)

      if (config.max_grad_norm > 0):
        grads, _ = tf.clip_by_global_norm(grads,config.max_grad_norm)

      self._lr = tf.Variable(0.0, trainable=False)
      optimizer = None
      args = config.optimizer_params
      if hasattr(tf.train,config.optimizer):
        optimizer = getattr(tf.train, config.optimizer)(learning_rate=self._lr,**args)
      else:
        raise RuntimeError("Unknown optimizer = %s"%config.optimizer)
     
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

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

  def _batch_relu_layer(self, x, size, scope):
    with tf.variable_scope(scope):
      h1 = tf.contrib.layers.fully_connected(x, size,
                                             # activation_fn=None,
                                             scope='dense')
      h2 = tf.contrib.layers.batch_norm(h1, 
                                        center=True, scale=True,
                                        is_training=self._phase,
                                        scope='bn')
      return h2 # tf.nn.relu(h2, 'relu')


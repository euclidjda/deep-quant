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

from base_model import BaseModel

class DeepRnnModel(BaseModel):
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

      self._max_unrollings = max_unrollings = config.max_unrollings
      self._num_inputs = num_inputs = config.num_inputs
      self._num_outputs = num_outputs = config.num_outputs
      num_hidden = config.num_hidden

      # input/target normalization params
      self._center = tf.get_variable('center',shape=[num_inputs],trainable=False)
      self._scale  = tf.get_variable('scale',shape=[num_inputs],trainable=False)
      
      batch_size = self._batch_size = tf.placeholder(tf.int32, shape=[])
      self._seq_lengths = tf.placeholder(tf.int32, shape=[None])
      self._keep_prob = tf.placeholder(tf.float32, shape=[])
      self._phase = tf.placeholder(tf.bool, name='phase')

      self._inputs = list()
      self._targets = list()
 
      for _ in range(max_unrollings):
        inp = tf.placeholder(tf.float32, shape=[None,num_inputs])
        tar =  tf.placeholder(tf.float32, shape=[None,num_outputs])
        self._inputs.append( inp )
        self._targets.append( tar )

      self._scaled_inputs = [None]*max_unrollings
      self._scaled_targets = [None]*max_unrollings
        
      for i in range(max_unrollings):
        if config.data_scaler is not None:
          self._scaled_inputs[i] = self._center_and_scale( self._inputs[i] )
        else:
          self._scaled_inputs[i] = self._inputs[i]
        if config.data_scaler is not None and config.scale_targets is True:
          self._scaled_targets[i] = self._center_and_scale( self._targets[i] )
        else:
          self._scaled_targets[i] = self._targets[i]
            
      hkp = self._keep_prob if config.hidden_dropout is True else 1.0
      ikp = self._keep_prob if config.input_dropout is True else 1.0
      rkp = self._keep_prob if config.rnn_dropout is True else 1.0
      
      def rnn_cell():
        cell = None
        if config.rnn_cell == 'gru':
          cell = tf.contrib.rnn.GRUCell(num_hidden)
        elif config.rnn_cell == 'lstm':
          cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_hidden,
                                                       dropout_keep_prob=rkp,
                                                       dropout_prob_seed=config.seed)
        assert(cell is not None)
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             output_keep_prob=hkp,
                                             input_keep_prob=ikp,seed=config.seed)
        return cell

      stacked_rnn = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(config.num_layers)])
      
      rnn_outputs, state = tf.contrib.rnn.static_rnn(stacked_rnn,
                                                     self._scaled_inputs, 
                                                     dtype=tf.float32,
                                                     sequence_length=self._seq_lengths)

      self._w = output_w = tf.get_variable("output_w", [num_hidden, num_outputs])
      output_b = tf.get_variable("output_b", [num_outputs])

      self._outputs = list()
      for i in range(max_unrollings):
        self._outputs.append( tf.nn.xw_plus_b( rnn_outputs[i], output_w, output_b ) )

      #last_target = self._scaled_targets[-1]
      #last_output = self._outputs[-1]

      last_output = tf.unstack(
        tf.reverse_sequence(
          tf.reshape(
            tf.concat(self._outputs, 1),[batch_size,max_unrollings,num_outputs] ), 
          self._seq_lengths,seq_axis=1,batch_axis=0),axis=1)[0]

      last_target = tf.unstack(
        tf.reverse_sequence(
          tf.reshape(
            tf.concat(self._scaled_targets, 1),[batch_size,max_unrollings,num_outputs] ), 
          self._seq_lengths,seq_axis=1,batch_axis=0),axis=1)[0]

      ktidx = config.target_idx
      
      self._o = self._reverse_center_and_scale( last_output )
      self._t = self._reverse_center_and_scale( last_target )
      self._o1 = self._o[:,ktidx]
      self._t1 = self._t[:,ktidx]

      seq_mask = tf.sequence_mask(self._seq_lengths*num_outputs, 
                                  max_unrollings*num_outputs, dtype=tf.float32)

      # self._inps = tf.multiply(seq_mask,tf.concat(self._scaled_inputs, 1))
      self._outs = outputs = tf.multiply(seq_mask,tf.concat(self._outputs, 1))
      self._tars = targets = tf.multiply(seq_mask,tf.concat(self._scaled_targets, 1))
      # outputs = tf.concat(self._outputs, 0)
      # targets = tf.concat(self._scaled_targets, 0)

      self._mse_all_steps = tf.losses.mean_squared_error(targets, outputs)
      self._mse_last_step = tf.losses.mean_squared_error(last_target, last_output)
      self._mse = tf.losses.mean_squared_error(last_target[:,ktidx], last_output[:,ktidx])

      if config.data_scaler is not None and config.scale_targets is True:
        self._predictions = self._reverse_center_and_scale( last_output )
      else:
        self._predictions = last_output
 
      # here is the learning part of the graph
      p1 = config.target_lambda
      p2 = config.rnn_lambda
      loss = p1 * self._mse + (1.0-p1)*(p2*self._mse_last_step + (1.0-p2)*self._mse_all_steps)
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss,tvars)

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

    

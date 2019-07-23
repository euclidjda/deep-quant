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

from models.base_model import BaseModel
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

class DeepCNNModel(BaseModel):
    """
    A Deep CNN regression model with an
    arbitrary number of convolutional hidden 
    layers and fully connected fixed width hidden layers.
    """
    def __init__(self, config):
        """
        Initialize the model
        Args:
          config
        """

        if hasattr(tf.nn,config.activation_fn):
            self._activation_fn = getattr(tf.nn, config.activation_fn)
        else:
            raise RuntimeError("Unknown activation function = %s"%self._activation_fn)
        
        self._max_unrollings = max_unrollings = config.max_unrollings
        self._num_inputs = num_inputs = config.num_inputs
        self._num_outputs = num_outputs = config.num_outputs

        # ---additional variables for CNN architecture---
        self._num_filters = num_filters = config.num_filters
        self._pool_size  = config.pool_size
        pool_sz = (self._pool_size, self._pool_size)
        self._conv_size = config.conv_size
        conv_size = (self._conv_size, self._conv_size)
        self._pooling = pool = config.pooling
        self._conv_blocks = conv_blocks = config.conv_blocks
        self._num_layers = fc_blocks = config.num_layers
        self._num_hidden = fc_layer_size = config.num_hidden
        # ---
        
        total_input_size = max_unrollings * num_inputs

        # input/target normalization params
        self._center = tf.get_variable('center',shape=[num_inputs],trainable=False)
        self._scale  = tf.get_variable('scale',shape=[num_inputs],trainable=False)

        batch_size = self._batch_size = tf.placeholder(tf.int32, shape=[])
        self._seq_lengths = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32, shape=[])
        self._phase = tf.placeholder(tf.bool, name='phase')

        self._inputs = list()
        self._targets = list()

        for _ in range(max_unrollings):
            self._inputs.append( tf.placeholder(tf.float32,
                                                  shape=[None,num_inputs]) )
            self._targets.append( tf.placeholder(tf.float32,
                                                  shape=[None,num_outputs]) )

        inputs = tf.reverse_sequence(tf.concat( self._inputs, 1 ),
                                     self._seq_lengths*num_inputs,
                                     seq_axis=1,batch_axis=0)
        # inputs = tf.concat( self._inputs, 1 )

        targets = tf.unstack(tf.reverse_sequence(tf.reshape(
          tf.concat(self._targets, 1),[batch_size,max_unrollings,num_outputs]),
          self._seq_lengths,seq_axis=1,batch_axis=0),axis=1)[0]
        # targets = self._targets[-1]

        # center and scale
        if config.data_scaler is not None:
            inputs = tf.divide(inputs - tf.tile(self._center,[max_unrollings]),
                              tf.tile(self._scale,[max_unrollings]))
            if config.scale_targets is True:
                targets = self._center_and_scale( targets )

        if config.input_dropout is True:
            inputs = self._input_dropout(inputs)

        num_prev = total_input_size

        seq_mask = tf.sequence_mask(self._seq_lengths*num_inputs,
                                    total_input_size, dtype=tf.float32)
        inputs = tf.multiply(seq_mask, inputs)
        outputs = inputs

        # ---Conv block portion---

        # Temp variables for CNN architecture
        x_dim = self._max_unrollings
        y_dim = self._num_inputs
        in_channels = 1
        out_channels = 1 * num_filters

        for i in range(conv_blocks):
            outputs = self._batch_conv_block(outputs, "layer_%d"%i, x_dim, y_dim, in_channels, out_channels, conv_size, pooling=pool, pool_size=pool_sz)

            #adjust dims for pooling
            if pool:
                x_dim = int(x_dim/pool_sz[0])
                y_dim = int(y_dim/pool_sz[1])

            #adjust in_channels and out_channels by mult by num filters
            in_channels = in_channels * num_filters 
            out_channels = out_channels * num_filters
            if config.hidden_dropout is True:
                outputs = tf.nn.dropout(outputs, self._keep_prob)
            num_prev = config.num_hidden

        # ---

        
        # Fully connected portion
        size = x_dim * y_dim * in_channels
        for i in range(fc_blocks):
            outputs = self._batch_relu_layer(outputs, "layer_%d"%i, size, fc_layer_size)
            if config.hidden_dropout is True:
                outputs = tf.nn.dropout(outputs, self._keep_prob)
            size = fc_layer_size
            #update fc_layer_size here if desired

        if config.skip_connections is True:
            num_prev = num_inputs+num_prev
            # skip_inputs = tf.slice(inputs, [0, 0], [batch_size, num_inputs] )
            skip_inputs = inputs[:,:num_inputs]
            outputs  = tf.concat( [ skip_inputs, outputs], 1)

        # final regression layer
        linear_b = tf.get_variable("linear_b", [num_outputs])
        linear_w = tf.get_variable("linear_w", [num_prev, num_outputs])
        outputs = tf.nn.xw_plus_b(outputs, linear_w, linear_b)

        if config.direct_connections is True:
            outputs = outputs + inputs[:,:num_outputs]

        self._inps = inputs
        self._tars = targets
        self._outs = outputs

        ktidx = config.target_idx
        self._mse = tf.losses.mean_squared_error(targets[:,ktidx], outputs[:,ktidx])
        self._mse_all = tf.losses.mean_squared_error(targets, outputs)

        if config.data_scaler is not None and config.scale_targets is True:
            self._predictions = self._reverse_center_and_scale( outputs )
        else:
            self._predictions = outputs

        # from here down is the learning part of the graph
        L = config.target_lambda
        loss = L * self._mse + (1.0 - L) * self._mse_all
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss,tvars)

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
        random_tensor = tf.tile(random_tensor,[1,self._max_unrollings])
        binary_tensor = math_ops.floor(random_tensor)

        ret = math_ops.div(inputs, self._keep_prob) * binary_tensor
        ret.set_shape(inputs.get_shape())

        return ret

    def _batch_relu_layer(self, x, scope, in_size, out_size):
        with tf.variable_scope(scope):
            x = tf.reshape(x, [self._batch_size, in_size])
            h1 = tf.contrib.layers.fully_connected(x, out_size,
                                                   activation_fn=None,
                                                   weights_regularizer=None)
            h2 = tf.contrib.layers.batch_norm(h1,
                                              center=True, scale=True,
                                              is_training=self._phase,
                                              scope='bn')
            # return tf.nn.relu(h2, 'relu')
            return self._activation_fn(h2,name='activation_fn')

    def _batch_conv_block(self, x, scope, x_dim, y_dim, in_channels, out_channels, conv_size = (3,3), pooling=False, pool_size=(2,2)):
        with tf.variable_scope(scope):
            
            x = tf.reshape(x, [self._batch_size, x_dim, y_dim, in_channels])
            h1 = tf.contrib.layers.conv2d(x, out_channels,
                                          conv_size,
                                          activation_fn=None,
                                          weights_regularizer=None)
            h2 = tf.contrib.layers.batch_norm(h1,
                                              center=True, scale=True,
                                              is_training=self._phase)

            # return tf.nn.relu(h2, 'relu')
            h3 = self._activation_fn(h2,name='activation_fn')

            #h3 = tf.layers.flatten(h3)

            #return h3
            if pooling:
                h3 = tf.contrib.layers.max_pool2d(h3, pool_size)

            #h4 = tf.layers.flatten(h3)

            return h3

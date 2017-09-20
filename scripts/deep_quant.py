#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''

# #! /usr/bin/env python3
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

import time
import os
import sys
import copy

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from batch_generator import BatchGenerator

#import model_utils
import configs as configs
from train import train_model

def get_configs():
  """Defines all configuration params passable to command line.
  """
  configs.DEFINE_string("datafile",'data.dat',"a datafile name.")
  configs.DEFINE_string("default_gpu",'',"The default GPU to use e.g., /gpu:0")
  configs.DEFINE_string("nn_type",'DeepRnnModel',"Model type")
  configs.DEFINE_string("key_field", 'gvkey',"Key column name header in datafile")
  configs.DEFINE_string("target_field", 'target',"Target column name header in datafile")
  configs.DEFINE_string("data_dir",'',"The data directory")
  configs.DEFINE_string("model_dir",'',"Model directory")
  configs.DEFINE_integer("num_unrollings",4,"Number of unrolling steps")
  configs.DEFINE_integer("stride",1,"How many steps to skip per unrolling")
  configs.DEFINE_integer("batch_size",1,"Size of each batch")
  configs.DEFINE_integer("num_layers",1, "Numer of RNN layers")
  configs.DEFINE_integer("num_inputs",10,"Number of inputs")
  configs.DEFINE_integer("num_outputs",2,"Number of classes (outputs)")
  configs.DEFINE_integer("num_hidden",10,"Number of hidden layer units")
  configs.DEFINE_float("init_scale",0.1, "Initial scale for weights")
  configs.DEFINE_float("max_grad_norm",10.0,"Gradient clipping")
  configs.DEFINE_integer("end_date",210001,"Last date to train on as YYYYMM")
  configs.DEFINE_float("keep_prob",1.0,"Keep probability for dropout")
  configs.DEFINE_boolean("train",True,"Train model otherwise inference only")
  configs.DEFINE_boolean("input_dropout",False,"Do dropout on input layer")
  configs.DEFINE_boolean("hidden_dropout",True,"Do dropout on hidden layers")
  configs.DEFINE_boolean("skip_connections",False,"Have direct connections between input and output in MLP")
  configs.DEFINE_boolean("use_cache",True,"Load data for logreg from cache (vs processing from batch generator)")
  configs.DEFINE_string("optimizer", 'GradientDescentOptimizer', 'Any tensorflow optimizer in tf.train')
  configs.DEFINE_float("lr_decay",0.9, "Learning rate decay")
  configs.DEFINE_float("initial_learning_rate",1.0,"Initial learning rate")
  configs.DEFINE_float("validation_size",0.0,"Size of validation set as %")
  configs.DEFINE_float("passes",1.0,"Passes through day per epoch")
  configs.DEFINE_float("rnn_loss_weight",None,"How much moret to weight kth example")
  configs.DEFINE_integer("max_epoch",0,"Stop after max_epochs")
  configs.DEFINE_integer("early_stop",None,"Early stop parameter")
  configs.DEFINE_integer("seed",None,"Seed for deterministic training")

  c = configs.ConfigValues()
  # Check to see if we are in training or testing mode

  return c

def main(_):
    config = get_configs()

    if config.train is True:
      train_model(config)
        
if __name__ == "__main__":
  tf.app.run()


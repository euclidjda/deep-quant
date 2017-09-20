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
import regex as re

from tensorflow.python.platform import gfile
from batch_generator import BatchGenerator

import model_utils

def pretty_progress(step, prog_int, dot_count):
  if ( (prog_int<=1) or (step % (int(prog_int)+1)) == 0):
    dot_count += 1; print('.',end=''); sys.stdout.flush()
  return dot_count

def run_epoch(session, model, train_data, valid_data,
                keep_prob=1.0, passes=1.0, verbose=False):

  if not train_data.num_batches > 0:
    raise RuntimeError("batch_size*num_unrollings is larger "
                         "than the training set size.")

  start_time = time.time()
  train_mse = valid_mse = 0.0
  dot_count = 0
  train_steps = int(passes*train_data.num_batches)
  valid_steps = valid_data.num_batches
  total_steps = train_steps+valid_steps
  prog_int = total_steps/100 # progress interval for stdout

  train_data.rewind() # make sure we start a beggining
  valid_data.rewind() # make sure we start a beggining

  print("Steps: %d "%total_steps,end=' ')

  for step in range(train_steps):
    batch = train_data.next_batch()
    train_mse += model.train_step(session, batch, keep_prob=keep_prob)
    if verbose: dot_count = pretty_progress(step,prog_int,dot_count)

  for step in range(valid_steps):
    batch = valid_data.next_batch()
    valid_mse += model.step(session, batch)
    if verbose: dot_count = pretty_progress(train_steps+step,prog_int,dot_count)
      
  # evaluate validation data

  if verbose:
    print("."*(100-dot_count),end='')
    print(" passes: %.2f  "
          "speed: %.0f seconds" % (passes,(time.time() - start_time)) )
  sys.stdout.flush()

  return (train_mse/train_steps,valid_mse/valid_steps)

def train_model(config):

  train_path = model_utils.get_data_path(config.data_dir,config.datafile)

  print("Loading training data ...")

  batches = BatchGenerator(train_path,config)

  train_data = batches.train_batches()
  valid_data = batches.valid_batches()
  
  tf_config = tf.ConfigProto( allow_soft_placement=True  ,
                              log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    if config.seed is not None:
      tf.set_random_seed(config.seed)

    print("Constructing model ...")

    model = model_utils.get_model(session, config, verbose=True)

    if config.early_stop is not None:
      print("Training will early stop without "
        "improvement after %d epochs."%config.early_stop)
    
    train_history = list()
    valid_history = list()
    # This sets the initial learning rate tensor
    lr = model.assign_lr(session,config.initial_learning_rate)

    for i in range(config.max_epoch):

      (train_mse, valid_mse) = run_epoch(session, model, train_data, valid_data,
                                          keep_prob=config.keep_prob, passes=config.passes,
                                          verbose=True)
      print( ('Epoch: %6d Train MSE: %.6f Valid MSE: %.6f Learning rate: %.4f') %
            (i + 1, train_mse, valid_mse, lr) )
      sys.stdout.flush()

      train_history.append( train_mse )
      valid_history.append( valid_mse )
      
    if re.match("Gradient|Momentum",config.optimizer):
      lr = model_utils.adjust_learning_rate(session, model, lr, config.dlr_decay, train_history )

      if not os.path.exists(config.model_dir):
        print("Creating directory %s" % config.model_dir)
        os.mkdir(config.model_dir)

      chkpt_file_prefix = "training.ckpt"
      if model_utils.stop_training(config,valid_history,chkpt_file_prefix):
        print("Training stopped.")
        quit()
      else:
        checkpoint_path = os.path.join(config.model_dir, chkpt_file_prefix)
        tf.train.Saver().save(session, checkpoint_path, global_step=i)
      

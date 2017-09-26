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

import time
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from deep_mlp_model import DeepMlpModel
#from deep_rnn_model import DeepRnnModel
#from log_reg_model import LogRegModel

def get_data_path(data_dir, filename):
    """
    Construct the data path for the experiement. If DEEP_QUANT_ROOT is
    defined in the environment, then the data path is relative to it.

    Args:
      data_dir: the directory name where experimental data is held
      filename: the data file name
    Returns:
      If DEEP_QUANT_ROOT is defined, the fully qualified data path is returned
      Otherwise a path relative to the working directory is returned
    """
    path = os.path.join( data_dir, filename ) 
    if data_dir != '.' and 'DEEP_QUANT_ROOT' in os.environ:
        path = os.path.join(os.environ['DEEP_QUANT_ROOT'], path)
    return path

def stop_training(config, perfs, file_prefix):
    """
    Early stop algorithm

    Args:
      config:
      perfs: History of validation performance on each iteration
      file_prefix: how to name the chkpnt file
    """
    window_size = config.early_stop
    if window_size is not None:
        if len(perfs) > window_size:
            total_min = min(perfs)
            window_min = min(perfs[-window_size:])
            # print("total_min=%.4f window_min=%.4f"%(total_min,window_min))
            if total_min < window_min:
                # early stop here
                best_idx = perfs.index(total_min) # index of total min
                chkpt_name = "%s-%d"%(file_prefix,best_idx)
                rewrite_chkpt(config.model_dir, chkpt_name)
                return True
    return False

def rewrite_chkpt(model_dir,chkpt_name):
    # open file model_dir/checkpoint
    path = model_dir+"/checkpoint"
    # write file as tensorflow expects
    with open(path, "w") as outfile:
      outfile.write("model_checkpoint_path: \"%s\"\n"%chkpt_name)
      outfile.write("all_model_checkpoint_paths: \"%s\"\n"%chkpt_name)



def adjust_learning_rate(session, model, 
                         learning_rate, lr_decay, cost_history, lookback=5):
  """
  Systematically decrease learning rate if current performance is not at
  least 1% better than the moving average performance

  Args:
    session: the current tf session for training
    model: the model being trained
    learning_rate: the current learning rate
    lr_decay: the learning rate decay factor
    cost_history: list of historical performance
  Returns:
    the updated learning rate being used by the model for training
  """
  lookback += 1
  if len(cost_history) >= lookback:
    mean = np.mean(cost_history[-lookback:-2])
    curr = cost_history[-1]
    # If performance has dropped by less than 1%, decay learning_rate
    if ((learning_rate >= 0.0001) and (mean > 0.0)
        and (mean >= curr) and (curr/mean >= 0.98)):
        learning_rate = learning_rate * lr_decay
  model.set_learning_rate(session, learning_rate)
  return learning_rate

def get_model(session, config, verbose=False):
    """
    Args:
      session: the tf session
      config: a config that specifies the model geometry and learning params
      verbose: print status output if true
    Returns:
      the model
    """
    if config.nn_type == 'logreg':
      model_file = os.path.join(config.model_dir, "logreg.pkl" )
      clf = LogRegModel(load_from=model_file)
      mtrain, mdeploy = clf, clf

    else:
      model = _create_model(session, config, verbose)

      ckpt = tf.train.get_checkpoint_state(config.model_dir)
      if ckpt and gfile.Exists(ckpt.model_checkpoint_path+".index"):
        if verbose:
          print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        tf.train.Saver(max_to_keep=200).restore(session,
                                                  ckpt.model_checkpoint_path)
      else:
        if verbose:
          print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    return model

def _create_model(session,config,verbose=False):
    """
    """
    all_objects = globals()
    if config.nn_type in all_objects:
        ModelConstructor = all_objects[config.nn_type]
    else:
      raise RuntimeError("Unknown net_type = %s"%config.nn_type)

    if verbose is True:
      print("Model has the following geometry:")
      print("  num_unroll  = %d"% config.num_unrollings)
      print("  batch_size  = %d"% config.batch_size)
      print("  num_inputs  = %d"% config.num_inputs)
      print("  num_hidden  = %d"% config.num_hidden)
      print("  num_output  = %d"% config.num_outputs)
      print("  num_layers  = %d"% config.num_layers)
      print("  optimizer   = %s"% config.optimizer)
      print("  device      = %s"% config.default_gpu)
    
    model = ModelConstructor(config)

    return model

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

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from models.deep_mlp_model import DeepMlpModel
from models.clvynt_model import ClvyntModel
from models.naive_model import NaiveModel
from models.deep_rnn_model import DeepRnnModel

#from log_reg_model import LogRegModel

def save_model(session, config, step):
    last_checkpoint_path = tf.train.latest_checkpoint(config.model_dir)
    checkpoint_path = os.path.join(config.model_dir, "training.ckpt")
    tf.train.Saver().save(session, checkpoint_path, global_step=step)
    if last_checkpoint_path is not None:
        os.remove(last_checkpoint_path+'.data-00000-of-00001')
        os.remove(last_checkpoint_path+'.index')
        os.remove(last_checkpoint_path+'.meta')

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
        start_time = time.time()
        if ckpt and gfile.Exists(ckpt.model_checkpoint_path+".index"):
            if verbose:
                print("Reading model parameters from {}...".format(
                    ckpt.model_checkpoint_path), end=' ')
            tf.train.Saver(max_to_keep=200).restore(session,
                                                    ckpt.model_checkpoint_path)
            if verbose:
                print("done in %.2f seconds."%(time.time() - start_time))
        else:
            if verbose:
                print("Creating model with fresh parameters ...", end=' ')
            session.run(tf.global_variables_initializer())
            if verbose:
                print("done in %.2f seconds."%(time.time() - start_time))

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
        print("  model_type  = %s"% config.nn_type)
        print("  min_unroll  = %d"% config.min_unrollings)
        print("  max_unroll  = %d"% config.max_unrollings)
        print("  stride      = %d"% config.stride)
        print("  batch_size  = %d"% config.batch_size)
        print("  num_inputs  = %d"% config.num_inputs)
        print("  num_outputs = %d"% config.num_outputs)
        print("  num_hidden  = %d"% config.num_hidden)
        print("  num_layers  = %d"% config.num_layers)
        print("  optimizer   = %s"% config.optimizer)
        print("  device      = %s"% config.default_gpu)

    initer = tf.random_uniform_initializer(-config.init_scale,config.init_scale,seed=config.seed)

    with tf.variable_scope("model", reuse=None, initializer=initer), \
        tf.device(config.default_gpu):

        model = ModelConstructor(config)

    return model

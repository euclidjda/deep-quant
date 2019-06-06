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

import tensorflow as tf
import regex as re
import math

from utils import data_utils, model_utils


def pretty_progress(step, prog_int, dot_count):
    if ( (prog_int<=1) or (step % (int(prog_int)+1)) == 0):
        dot_count += 1; print('.',end=''); sys.stdout.flush()
    return dot_count


def run_epoch_mve(session, model, train_data, valid_data,
                keep_prob=1.0, passes=1.0, verbose=False):
    """Runs epoch for MVE type UQ Model"""

    if not train_data.num_batches > 0:
        raise RuntimeError("batch_size*max_unrollings is larger "
                             "than the training set size.")

    uq = True
    start_time = time.time()
    train_mse = train_mse_var = valid_mse = valid_mse_var = 0.0
    dot_count = 0
    train_steps = int(passes*train_data.num_batches)
    valid_steps = valid_data.num_batches
    total_steps = train_steps+valid_steps
    prog_int = total_steps/100 # progress interval for stdout

    train_data.shuffle() # we want to randomly shuffle train data
    valid_data.rewind()  # make sure we start a beginning

    print("Steps: %d "%total_steps,end=' ')

    for step in range(train_steps):
        batch = train_data.next_batch()
        step_mse = model.train_step(session, batch, keep_prob=keep_prob, uq=uq, UQ_model_type='MVE')
        train_mse += step_mse[0]
        train_mse_var += step_mse[1]

        if verbose: dot_count = pretty_progress(step,prog_int,dot_count)

    for step in range(valid_steps):
        batch = valid_data.next_batch()
        (mse, mse_var, _, _) = model.step(session, batch, uq=uq, UQ_model_type='MVE')
        valid_mse += mse
        valid_mse_var += mse_var
        if verbose: dot_count = pretty_progress(train_steps+step,prog_int,dot_count)

    # evaluate validation data

    if verbose:
        print("."*(100-dot_count),end='')
        print(" passes: %.2f  "
              "speed: %.0f seconds" % (passes, (time.time() - start_time)))
    sys.stdout.flush()

    return train_mse/train_steps, train_mse_var/train_steps, valid_mse/valid_steps, valid_mse_var/valid_steps


def run_epoch_pie(session, model, train_data, valid_data,
                keep_prob=1.0, passes=1.0, verbose=False):
    """Runs epoch for PIE type UQ Model"""

    if not train_data.num_batches > 0:
        raise RuntimeError("batch_size*max_unrollings is larger "
                             "than the training set size.")

    uq = True
    start_time = time.time()
    train_mpiw = train_picp = train_picp_loss = valid_mpiw = valid_picp = valid_picp_loss = 0.0
    dot_count = 0
    train_steps = int(passes*train_data.num_batches)
    valid_steps = valid_data.num_batches
    total_steps = train_steps+valid_steps
    prog_int = total_steps/100 # progress interval for stdout

    train_data.shuffle() # we want to randomly shuffle train data
    valid_data.rewind()  # make sure we start a beginning

    print("Steps: %d "%total_steps,end=' ')

    for step in range(train_steps):
        batch = train_data.next_batch()
        step_mpiw_picp = model.train_step(session, batch, keep_prob=keep_prob, uq=uq, UQ_model_type='PIE')
        train_mpiw += step_mpiw_picp[0]
        train_picp += step_mpiw_picp[2]
        train_picp_loss += step_mpiw_picp[1]

        if verbose: dot_count = pretty_progress(step, prog_int, dot_count)

    for step in range(valid_steps):
        batch = valid_data.next_batch()
        (mpiw, picp_loss, picp, _, _) = model.step(session, batch, uq=uq, UQ_model_type='PIE')
        valid_mpiw += mpiw
        valid_picp += picp
        valid_picp_loss += picp_loss
        if verbose: dot_count = pretty_progress(train_steps+step, prog_int, dot_count)

    # evaluate validation data

    if verbose:
        print("."*(100-dot_count),end='')
        print(" passes: %.2f  "
              "speed: %.0f seconds" % (passes, (time.time() - start_time)))
    sys.stdout.flush()

    return train_mpiw/train_steps, train_picp/train_steps, train_picp_loss/train_steps, \
           valid_mpiw/valid_steps, valid_picp/valid_steps, valid_picp_loss/valid_steps


def stop_training(config, perfs):
    """
    Early stop algorithm
    Args:
      config:
      perfs: History of validation performance on each iteration
      file_prefix: how to name the chkpnt file
    """
    window_size = config.early_stop
    if ( (window_size is not None)
     and (len(perfs) > window_size)
     and (min(perfs) < min(perfs[-window_size:])) ):
        return True
    elif config.train_until > perfs[-1]:
        return True
    else:
        return False


def train_model(config):
    if config.start_date is not None:
        print("Training start date: ", config.start_date)
    if config.start_date is not None:
        print("Training end date: ", config.end_date)

    print("Loading training data from %s ..."%config.datafile)
    train_data = None
    valid_data = None

    if (config.validation_size > 0.0) or (config.split_date is not None):
        train_data, valid_data = data_utils.load_train_valid_data(config)
    else:
        train_data = data_utils.load_all_data(config, is_training_only=True)
        valid_data = train_data
        
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:
        if config.seed is not None:
            tf.set_random_seed(config.seed)

        print("Constructing model ...")
        model = model_utils.get_model(session, config, verbose=True)

        if config.data_scaler is not None:
            start_time = time.time()
            print("Calculating scaling parameters ...", end=' '); sys.stdout.flush()
            scaling_params = train_data.get_scaling_params(config.data_scaler)
            model.set_scaling_params(session,**scaling_params)
            print("done in %.2f seconds."%(time.time() - start_time))
            print("%-10s %-6s %-6s"%('feature','mean','std'))
            for i in range(len(train_data.feature_names)):
                center = "%.4f"%scaling_params['center'][i];
                scale  = "%.4f"%scaling_params['scale'][i];
                print("%-10s %-6s %-6s"%(train_data.feature_names[i],
                                         center,scale))
            sys.stdout.flush()

        if config.early_stop is not None:
            print("Training will early stop without "
              "improvement after %d epochs."%config.early_stop)

        train_history = list()
        valid_history = list()

        lr = model.set_learning_rate(session, config.learning_rate)

        train_data.cache(verbose=True)
        valid_data.cache(verbose=True)

        for i in range(config.max_epoch):

            # MVE Epoch
            if config.UQ_model_type == 'MVE':
                (train_mse, train_mse_var, valid_mse, valid_mse_var) = run_epoch_mve(session, model, train_data,
                                                                                     valid_data,
                                                                                     keep_prob=config.keep_prob,
                                                                                     passes=config.passes,
                                                                                     verbose=True)
                # Status to check if valid mse is nan, used to stop training
                if math.isnan(valid_mse):
                    is_metric_nan = True
                else:
                    is_metric_nan = False
                print('Epoch: %d Train MSE: %.8f Valid MSE: %.8f Learning rate: %.4f' %
                      (i + 1, train_mse, valid_mse, lr))
                print('Epoch: %d Train MSE_w_variance: %.8f Valid MSE_w_variance: %.8f Learning rate: %.4f' %
                      (i + 1, train_mse_var, valid_mse_var, lr))
                sys.stdout.flush()

                train_history.append(train_mse_var)
                valid_history.append(valid_mse_var)

            # PIE Epoch
            elif config.UQ_model_type == 'PIE':
                (train_mpiw, train_picp, train_picp_loss, valid_mpiw, valid_picp, valid_picp_loss) = \
                    run_epoch_pie(session, model, train_data, valid_data,
                                  keep_prob=config.keep_prob,
                                  passes=config.passes,
                                  verbose=True)

                train_loss = train_mpiw + config.picp_lambda*train_picp_loss
                valid_loss = valid_mpiw + config.picp_lambda*valid_picp_loss
                # Status to check if valid loss is nan, used to stop training
                if math.isnan(valid_loss):
                    is_metric_nan = True
                else:
                    is_metric_nan = False

                print('Epoch: %d Train MPIW: %.8f Valid MPIW: %.8f Learning rate: %.4f' %
                      (i + 1, train_mpiw, valid_mpiw, lr))
                print('Epoch: %d Train PICP: %.8f Valid PICP: %.8f' %
                      (i + 1, train_picp, valid_picp))
                print('Epoch: %d Train LOSS: %.8f Valid LOSS: %.8f' %
                      (i + 1, train_loss, valid_loss ))

                sys.stdout.flush()

                train_history.append(train_loss)
                valid_history.append(valid_loss)

            if re.match("Gradient|Momentum", config.optimizer):
                lr = model_utils.adjust_learning_rate(session, model, 
                                                      lr, config.lr_decay, train_history)

            if not os.path.exists(config.model_dir):
                print("Creating directory %s" % config.model_dir)
                os.mkdir(config.model_dir)

            if is_metric_nan:
                print("Training failed due to nan.")
                quit()
            elif stop_training(config, valid_history):
                print("Training stopped.")
                quit()
            else:
                if ( (config.early_stop is None) or 
                     (valid_history[-1] <= min(valid_history)) ):
                    model_utils.save_model(session, config, i)

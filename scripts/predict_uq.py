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
import math

import numpy as np
import tensorflow as tf
import regex as re
import pandas as pd

from tensorflow.python.platform import gfile
from batch_generator import BatchGenerator

from utils import model_utils,data_utils
import utils


def print_vector(name,v):
    print("%s: "%name,end='')
    for i in range(len(v)):
        print("%.2f "%v[i],end=' ')
    print()


def predict(config):
    if config.UQ_model_type == 'MVE':
        predict_mve(config)
    elif config.UQ_model_type == 'PIE':
        predict_pie(config)


def predict_mve(config):
    datafile = config.datafile

    if config.predict_datafile is not None:
        datafile = config.predict_datafile

    print("Loading data from %s ..."%datafile)
    path = utils.data_utils.get_data_path(config.data_dir,datafile)

    config.batch_size = 1
    batches = BatchGenerator(path, 
                             config,
                             require_targets=config.require_targets, 
                             verbose=True)
    batches.cache(verbose=True)

    tf_config = tf.ConfigProto( allow_soft_placement=True  ,
                                log_device_placement=False )
    tf_config.gpu_options.allow_growth = True

    # Initialize DataFrames
    df_target = pd.DataFrame()
    df_output = pd.DataFrame()
    df_variance = pd.DataFrame()
    df_mse = pd.DataFrame()
    df_mse_var = pd.DataFrame()

    df_list = [df_target, df_output, df_variance, df_mse, df_mse_var]

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        model = model_utils.get_model(session, config, verbose=True)

        perfs = dict()
        perfs_p = dict()

        for i in range(batches.num_batches):
            batch = batches.next_batch()

            (mse, mse_var, preds, preds_variance) = model.step(session, batch, keep_prob=config.keep_prob_pred,
                                                               uq=config.UQ, UQ_model_type='MVE')
            # (mse, preds) = model.debug_step(session, batch)

            date = batch_to_date(batch)
            key = batch_to_key(batch)
            if math.isnan(mse) is False:
                if date not in perfs:
                    perfs[date] = list()
                    perfs_p[date] = list()
                perfs[date].append(mse)
                perfs_p[date].append(mse_var)

            # Print according to the options
            if config.pretty_print_preds:
                pretty_print_predictions(batches, batch, preds, preds_variance,  mse, mse_var)
            elif config.print_preds:
                print_predictions(config, batches, batch, preds, preds_variance,  mse, mse_var)

            # Get values and update DataFrames if df_dirname is provided in config
            if config.df_dirname is not None:
                # Get all the values
                target_val = get_value(batches, batch, 'target')
                output_val = get_value(batches, batch, 'output', preds)
                variance_val = get_value(batches, batch, 'variance', preds_variance)
                mse_val = mse
                mse_var_val = mse_var
                values_list = [target_val, output_val, variance_val, mse_val, mse_var_val]

                # Update DataFrames
                for j in range(len(df_list)):
                    assert(len(df_list) == len(values_list))
                    df_list[j] = update_df(df_list[j], date, key, values_list[j])

        # Save the DataFrames
        if config.df_dirname:
            if not os.path.isdir(config.df_dirname):
                os.makedirs(config.df_dirname)
            save_names = ['target-df.pkl', 'output-df.pkl', 'variance-df.pkl', 'mse-df.pkl', 'mse-var-df.pkl']

            for j in range(len(df_list)):
                assert(len(df_list) == len(save_names))
                df_list[j].to_pickle(os.path.join(config.df_dirname, save_names[j]))

        # MSE Outfile
        if config.mse_outfile is not None:
            with open(config.mse_outfile, "w") as f:
                for date in sorted(perfs):
                    mean = np.mean(perfs[date])
                    print("%s %.6f %d"%(date, mean, len(perfs[date])), file=f)
                total_mean = np.mean( [x for v in perfs.values() for x in v] )
                print("Total %.6f" % total_mean, file=f)
            f.closed
        else:
            exit()

        # MSE with variance outfile
        if config.mse_var_outfile is not None:
            with open(config.mse_var_outfile, "w") as f:
                for date in sorted(perfs_p):
                    mean = np.mean(perfs_p[date])
                    print("%s %.6f %d"%(date, mean, len(perfs_p[date])), file=f)
                total_mean = np.mean( [x for v in perfs_p.values() for x in v] )
                print("Total %.6f" % total_mean,file=f)
            f.closed
        else:
            exit()


def predict_pie(config):
    """ Doesn't use print options. Only outputs dataframes"""
    datafile = config.datafile

    if config.predict_datafile is not None:
        datafile = config.predict_datafile

    print("Loading data from %s ..."%datafile)
    path = utils.data_utils.get_data_path(config.data_dir,datafile)

    config.batch_size = 1
    batches = BatchGenerator(path,
                             config,
                             require_targets=config.require_targets,
                             verbose=True)
    batches.cache(verbose=True)

    tf_config = tf.ConfigProto( allow_soft_placement=True  ,
                                log_device_placement=False )

    # Initialize DataFrames
    df_target = pd.DataFrame()
    df_output_lb = pd.DataFrame()
    df_output_ub = pd.DataFrame()

    df_list = [df_target, df_output_lb, df_output_ub]

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        model = model_utils.get_model(session, config, verbose=True)

        for i in range(batches.num_batches):
            batch = batches.next_batch()

            (mpiw, _, _, preds_lb, preds_ub) = model.step(session, batch, keep_prob=config.keep_prob_pred,
                                                          uq=config.UQ, UQ_model_type='PIE')
            # (mse, preds) = model.debug_step(session, batch)

            date = batch_to_date(batch)
            key = batch_to_key(batch)

            # Dummy input to be consistent with the rest of the predictions printing options. MSE = 0.0. It is not
            # evaluated in PIE case
            mse_dummy = mse_var_dummy = 0.0

            # Print every n iterations to check the progress for monitoring
            if i % 10000 == 0:
                pretty_print_predictions( batches, batch, preds_lb, preds_ub, mse_dummy, mse_var_dummy)

            # Get values and update DataFrames if df_dirname is provided in config
            if config.df_dirname is not None:
                # Get all values
                target_val = get_value(batches, batch, 'target')
                output_lb_val = get_value(batches, batch, 'output_lb', preds_lb)
                output_ub_val = get_value(batches, batch, 'output_ub', preds_ub)
                values_list = [target_val, output_lb_val, output_ub_val]

                # Update DataFrames
                for j in range(len(df_list)):
                    assert(len(df_list) == len(values_list))
                    df_list[j] = update_df(df_list[j], date, key, values_list[j])

        # Save the DataFrames
        if not os.path.isdir(config.df_dirname):
            os.makedirs(config.df_dirname)
        save_names = ['target-df.pkl', 'output-lb-df.pkl', 'output-ub-df.pkl']

        for j in range(len(df_list)):
            assert(len(df_list) == len(save_names))
            df_list[j].to_pickle(os.path.join(config.df_dirname, save_names[j]))
    return


def batch_to_key(batch):
    idx = batch.seq_lengths[0]-1
    assert(0 <= idx)
    assert(idx < len(batch.attribs))
    return batch.attribs[idx][0][0]


def batch_to_date(batch):
    idx = batch.seq_lengths[0]-1
    assert(0 <= idx)
    assert(idx < len(batch.attribs))
    if (batch.attribs[idx][0] is None):
        print(idx)
        exit()
    return batch.attribs[idx][0][1]


def pretty_print_predictions(batches, batch, preds, preds_variances, mse, mse_var):
    key     = batch_to_key(batch)
    date    = batch_to_date(batch)

    L = batch.seq_lengths[0]
    targets = batch.targets[L-1][0]
    outputs = preds[0]
    variances = preds_variances[0]
    # variances = np.exp(-1*variances) # for precision formulation

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)

    print("%s %s mse=%.8f mse_var=%.8f"%(date, key, mse, mse_var))
    inputs = batch.inputs
    for i in range(L):
        print_vector("input[t-%d]"%(L-i-1), batches.get_raw_inputs(batch, 0, inputs[i][0]))
    print_vector("output[t+1]", batches.get_raw_outputs(batch, 0, outputs))
    print_vector("target[t+1]", batches.get_raw_outputs(batch, 0, targets))
    print_vector("variance[t+1]", batches.get_raw_outputs(batch, 0, variances))

    print("--------------------------------")
    sys.stdout.flush()


def print_predictions(config, batches, batch, preds, preds_variances, mse, mse_var):
    key     = batch_to_key(batch)
    date    = batch_to_date(batch)
    inputs  = batch.inputs[-1][0]
    outputs = preds[0]
    variances = preds_variances[0]

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)

    # Raw outputs
    out = batches.get_raw_outputs(batch, 0, outputs)
    prec = batches.get_raw_outputs(batch, 0, variances)

    if config.print_normalized_outputs:
        out_str = 'out ' + ' '.join(["%.3f" % outputs[i] for i in range(len(outputs))])
        prec_str = 'var ' + ' '.join(["%.3f" % variances[i] for i in range(len(variances))])
    else:
        out_str = 'out ' + ' '.join(["%.3f"%out[i] for i in range(len(out))])
        prec_str = 'var ' + ' '.join(["%.3f" % prec[i] for i in range(len(prec))])

    print("%s %s %s %s"%(date, key, out_str, str(mse)))
    print("%s %s %s %s" % (date, key, prec_str, str(mse_var)))

    sys.stdout.flush()


def update_df(df, date, key, value):
    """
    Updates the dataframe with key as column, date as index
    :param df:  Dataframe to be updated
    :param date: date
    :param key: gvkey
    :param value: value to be inserted
    :return: updated df
    """
    date = pd.to_datetime(date, format="%Y%m")
    df.loc[date, key] = value
    return df


def get_value(batches, batch, field, predictions=None, output_field=3):
    """
    Extracts the appropriate field value from batch or predictions
    :param batches:
    :param batch: batch
    :param field: field
    :param predictions: predictions eg outputs, variances
    :param output_field: field to be extracted
    :return: value from batch or mse value
    """

    assert(field in ['target', 'output', 'variance', 'output_lb', 'output_ub'])

    if field == 'target':
        l = batch.seq_lengths[0]
        targets = batch.targets[l - 1][0]
        value = batches.get_raw_outputs(batch, 0, targets)[output_field]
    else:
        value = batches.get_raw_outputs(batch, 0, predictions[0])[output_field]

    return value




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

import tensorflow as tf

import configs as configs
from train import train_model
from predict import predict


def get_configs():
    """
    Defines all configuration params passable to command line.
    """
    configs.DEFINE_string("name",'none',"A name for the config.")
    configs.DEFINE_string("datafile", 'open_dataset.dat', "a datafile name.")
    configs.DEFINE_string("predict_datafile", None, "If predict_datafile is not None, use it instead of datafile for predictions")
    configs.DEFINE_string("mse_outfile", None, "A file to write mse values during predict phase.")
    configs.DEFINE_string("scalesfile", None, "Optional file for storing scaling params")
    configs.DEFINE_string("default_gpu", '', "The default GPU to use e.g., /gpu:0")
    configs.DEFINE_string("nn_type",'DeepRnnModel',"Model type")
    configs.DEFINE_string("active_field", 'active', "Key column name header for active indicator")
    configs.DEFINE_string("date_field", 'date', "Name of data column.")
    configs.DEFINE_string("key_field", 'gvkey',"Key column name header in datafile")
    configs.DEFINE_string("target_field", 'oiadpq_ttm',"Target column name header in datafile")
    configs.DEFINE_string("scale_field", 'mrkcap',"Feature to scale inputs by")
    configs.DEFINE_string("financial_fields", '',"Shared input and target field names")
    configs.DEFINE_string("aux_fields", None,"non-target, input only fields")
    configs.DEFINE_string("dont_scale", None,"Names of fields to not scale")
    configs.DEFINE_string("data_dir",'',"The data directory")
    configs.DEFINE_string("model_dir",'chkpts',"Model (checkpoint) directory")
    configs.DEFINE_string("rnn_cell",'gru',"lstm or gru")
    configs.DEFINE_string("activation_fn",'relu',"MLP activation function in tf.nn.*")
    configs.DEFINE_integer("num_inputs", -1,"")
    configs.DEFINE_integer("num_outputs", -1,"")
    configs.DEFINE_integer("target_idx",None,"")
    configs.DEFINE_integer("min_unrollings",None,"Min number of unrolling steps")
    configs.DEFINE_integer("max_unrollings",None,"Max number of unrolling steps")
    configs.DEFINE_integer("min_years",None,"Alt to min_unrollings")
    configs.DEFINE_integer("max_years",None,"Alt to max_unrollings")
    configs.DEFINE_integer("pls_years",None,"Alt to max_years. max_years = min_year+pls_years")
    # num_unrollings is being depricated, replaced with max_unrollings
    configs.DEFINE_integer("num_unrollings",4,"Number of unrolling steps")
    configs.DEFINE_integer("stride",12,"How many steps to skip per unrolling")
    configs.DEFINE_integer("forecast_n",12,"How many steps to forecast into the future")
    configs.DEFINE_integer("batch_size",1,"Size of each batch")
    configs.DEFINE_integer("num_layers",1, "Numer of RNN layers")
    configs.DEFINE_integer("num_hidden",10,"Number of hidden layer units")
    configs.DEFINE_float("training_noise",None, "Level of training noise as multiple of 1-stdev")
    configs.DEFINE_float("init_scale",0.1, "Initial scale for weights")
    configs.DEFINE_float("max_grad_norm",10.0,"Gradient clipping")
    configs.DEFINE_integer("start_date",None,"First date to train on as YYYYMM")
    configs.DEFINE_integer("end_date",None,"Last date to train on as YYYYMM")
    configs.DEFINE_integer("split_date",None,"Date to split train/test on.")
    configs.DEFINE_float("keep_prob",1.0,"Keep probability for dropout")
    configs.DEFINE_boolean("train",True,"Train model otherwise inference only")
    configs.DEFINE_boolean("require_targets",False,"Require target values for test predictions")
    configs.DEFINE_boolean("input_dropout",False,"Do dropout on input layer")
    configs.DEFINE_boolean("hidden_dropout",False,"Do dropout on hidden layers")
    configs.DEFINE_boolean("rnn_dropout",False,"Do dropout on recurrent connections")
    configs.DEFINE_boolean("skip_connections",False,"Have a linear fully connected weight skip hidden units in MLP")
    configs.DEFINE_boolean("direct_connections",False,"Have direct connections between input and output in MLP or RNN")
    configs.DEFINE_boolean("use_cache",True,"Load data for logreg from cache (vs processing from batch generator)")
    configs.DEFINE_boolean("pretty_print_preds",False,"Print predictions in tabular format with inputs, targets, and keys")
    configs.DEFINE_boolean("scale_targets",True,"")
    configs.DEFINE_boolean("backfill",False,"Backfill seq history to max_unrollings with data in first time step")
    configs.DEFINE_boolean("log_squasher",True,"Squash large normalized inputs with natural log function")
    configs.DEFINE_boolean("ts_smoother",False,"Use smoother on data time series during training")
    configs.DEFINE_string("data_scaler",None,'sklearn scaling algorithm or None if no scaling')
    configs.DEFINE_string("optimizer", 'GradientDescentOptimizer', 'Any tensorflow optimizer in tf.train')
    configs.DEFINE_string("optimizer_params",None, 'Additional optimizer params such as momentum')
    configs.DEFINE_float("learning_rate",0.6,"The initial starting learning rate")
    configs.DEFINE_float("lr_decay",0.9, "Learning rate decay")
    configs.DEFINE_float("validation_size",0.0,"Size of validation set as %, ie. .3 = 30% of data")
    configs.DEFINE_float("train_until",0.0,"Train until validation MSE is less than this value")
    configs.DEFINE_float("passes",1.0,"Passes through day per epoch")
    configs.DEFINE_float("target_lambda",0.5,"How much to weight last step vs. all steps in loss")
    configs.DEFINE_float("rnn_lambda",0.5,"How much to weight last step vs. all steps in loss")
    configs.DEFINE_float("l2_alpha",0.0,"L2 Regularization")
    configs.DEFINE_integer("max_epoch",0,"Stop after max_epochs")
    configs.DEFINE_integer("early_stop",None,"Early stop parameter")
    configs.DEFINE_integer("seed",None,"Seed for deterministic training")
    configs.DEFINE_integer("cache_id",None,"A unique experiment key for traking a cahce")

    c = configs.ConfigValues()

    if c.min_unrollings is None:
        c.min_unrollings = c.num_unrollings

    if c.max_unrollings is None:
        c.max_unrollings = c.num_unrollings

    if c.min_years is not None:
        c.min_unrollings = c.min_years * ( 12 // c.stride )
        if c.max_years is not None:
            c.max_unrollings = (c.max_years) * ( 12 // c.stride )
        elif c.pls_years is None:
            c.max_unrollings = c.min_unrollings
        else:
            c.max_unrollings = (c.min_years+c.pls_years) * ( 12 // c.stride )

    # optimizer_params is a string of the form "param1=value1,param2=value2,..."
    # this maps it to dictionary { param1 : value1, param2 : value2, ...}
    if c.optimizer_params is None:
        c.optimizer_params = dict()
    else:
        args_list = [p.split('=') for p in c.optimizer_params.split(',')]
        params = dict()
        for p in args_list:
            params[p[0]] = float(p[1])
        c.optimizer_params = params
        assert('learning_rate' not in c.optimizer_params)

    return c


def main(_):
    config = get_configs()

    # Check to see if we are in training or testing mode
    if config.train is True:
        train_model(config)
    else:
        predict(config)

if __name__ == "__main__":
    tf.app.run()

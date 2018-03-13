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

    path = utils.data_utils.get_data_path(config.data_dir,config.datafile)

    config.batch_size = 1
    batches = BatchGenerator(path, config,
                             require_targets=False, verbose=True)
    batches.cache(verbose=True)

    tf_config = tf.ConfigProto( allow_soft_placement=True  ,
                                log_device_placement=False )

    with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

        model = model_utils.get_model(session, config, verbose=True)

        perfs = dict()

        for i in range(batches.num_batches):
            batch = batches.next_batch()

            (mse, preds) = model.step(session, batch)
            # (mse, preds) = model.debug_step(session, batch)

            if math.isnan(mse) is False:
                date = batch_to_date(batch)
                if date not in perfs:
                    perfs[date] = list()
                perfs[date].append(mse)

            if config.pretty_print_preds is True:
                pretty_print_predictions(batches, batch, preds, mse)
            else:
                print_predictions(batches, batch, preds)

        if config.mse_outfile is not None:
            with open(config.mse_outfile,"w") as f:
                for date in sorted(perfs):
                    mean = np.mean(perfs[date])
                    print("%s %.6f %d"%(date,mean,len(perfs[date])),file=f)
                total_mean = np.mean( [x for v in perfs.values() for x in v] )
                print("Total %.6f"%(total_mean),file=f)
            f.closed
        else:
            exit()

def batch_to_key(batch):
    idx = batch.seq_lengths[0]-1
    assert( 0<= idx )
    assert( idx < len(batch.attribs) )
    return batch.attribs[idx][0][0]

def batch_to_date(batch):
    idx = batch.seq_lengths[0]-1
    assert( 0<= idx )
    assert( idx < len(batch.attribs) )
    if (batch.attribs[idx][0] is None):
        print(idx)
        exit()
    return batch.attribs[idx][0][1]

def pretty_print_predictions(batches, batch, preds, mse):
    key     = batch_to_key(batch)
    date    = batch_to_date(batch)

    L = batch.seq_lengths[0]
    targets = batch.targets[L-1][0]
    outputs = preds[0]

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)

    print("%s %s mse=%.4f"%(date,key,mse))
    inputs = batch.inputs
    for i in range(L):
        print_vector("input[t-%d]"%(L-i-1),batches.get_raw_inputs(batch,0,inputs[i][0]) )
    print_vector("output[t+1]", batches.get_raw_outputs(batch,0,outputs) )
    print_vector("target[t+1]", batches.get_raw_outputs(batch,0,targets) )
    print("--------------------------------")
    sys.stdout.flush()

def print_predictions(batches, batch, preds):
    key     = batch_to_key(batch)
    date    = batch_to_date(batch)
    inputs  = batch.inputs[-1][0]
    outputs = preds[0]

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    out = batches.get_raw_outputs(batch,0,outputs)
    out_str = ' '.join(["%.3f"%out[i] for i in range(len(out))])

    print("%s %s %s"%(date,key,out_str))
    sys.stdout.flush()

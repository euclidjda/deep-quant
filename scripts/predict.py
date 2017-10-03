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

def print_vector(name,v):
  print("%s: "%name,end='')
  for i in range(len(v)):
    print("%.2f "%v[i],end=' ')
  print()
            

def unlogmap(s,x):
  y = s * np.multiply(np.sign(x),np.expm1(np.fabs(x)))
  return y

def predict(config):

  path = model_utils.get_data_path(config.data_dir,config.datafile)

  config.batch_size = 1  
  batches = BatchGenerator(path,config)

  tf_config = tf.ConfigProto( allow_soft_placement=True  ,
                              log_device_placement=False )

  with tf.Graph().as_default(), tf.Session(config=tf_config) as session:

    model = model_utils.get_model(session, config, verbose=False)

    print("Num data points is %d"%batches.num_batches)
    
    for i in range(batches.num_batches):

      batch = batches.next_batch()

      (mse, preds) = model.step(session, batch)

      key     = batch.attribs[-1][0][0]
      date    = batch.attribs[-1][0][1]
      inputs  = batch.inputs[-1][0]
      targets = batch.targets[-1][0]
      outputs = preds[0]
      scale   = batch.seq_scales[0]
      
      np.set_printoptions(suppress=True)
      np.set_printoptions(precision=3)
      
      print("%s %s "%(key,date))
      print_vector("input[t-2]", unlogmap(scale, batch.inputs[-2][0]) )
      print_vector("input[t-1]", unlogmap(scale, batch.inputs[-1][0]) )
      print_vector("output[t ]", unlogmap(scale, outputs) )
      print_vector("target[t ]", unlogmap(scale, targets) )
      print("--------------------------------")

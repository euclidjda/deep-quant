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
import regex as re
import pandas as pd

import model_utils
import configs as configs
from batch_generator_test import BatchGenerator


def get_configs():
  """Defines all configuration params passable to command line.
  """
  configs.DEFINE_string("datafile",'data.dat',"a datafile name.")
  configs.DEFINE_string("mse_outfile",None,"A file write mse values during predict only phase.")
  configs.DEFINE_string("default_gpu",'',"The default GPU to use e.g., /gpu:0")
  configs.DEFINE_string("nn_type",'DeepRnnModel',"Model type")
  configs.DEFINE_string("active_field", 'active',"Key column name header for active indicator")
  configs.DEFINE_string("key_field", 'gvkey',"Key column name header in datafile")
  configs.DEFINE_string("target_field", 'oiadpq_ttm',"Target column name header in datafile")
  configs.DEFINE_string("scale_field", 'mrkcap',"Feature to scale inputs by")
  configs.DEFINE_string("first_feature_field", '',"First feature")
  configs.DEFINE_string("feature_fields", '',"shared input and target field names")
  configs.DEFINE_string("aux_input_fields", None,"non-target, input only fields")
  configs.DEFINE_string("data_dir",'',"The data directory")
  configs.DEFINE_string("model_dir",'',"Model directory")
  configs.DEFINE_string("rnn_cell",'gru',"lstm or gru")
  configs.DEFINE_integer("num_inputs", -1,"")
  configs.DEFINE_integer("num_outputs", -1,"")
  configs.DEFINE_integer("target_idx",None,"")
  configs.DEFINE_integer("num_unrollings",4,"Number of unrolling steps")
  configs.DEFINE_integer("predict_steps",1,"Average future preds over this many steps")
  configs.DEFINE_integer("stride",1,"How many steps to skip per unrolling")
  configs.DEFINE_integer("batch_size",1,"Size of each batch")
  configs.DEFINE_integer("num_layers",1, "Numer of RNN layers")
  configs.DEFINE_integer("num_hidden",10,"Number of hidden layer units")
  configs.DEFINE_float("init_scale",0.1, "Initial scale for weights")
  configs.DEFINE_float("max_grad_norm",10.0,"Gradient clipping")
  configs.DEFINE_integer("start_date",None,"First date to train on as YYYYMM")
  configs.DEFINE_integer("end_date",None,"Last date to train on as YYYYMM")
  configs.DEFINE_float("keep_prob",1.0,"Keep probability for dropout")
  configs.DEFINE_boolean("train",True,"Train model otherwise inference only")
  configs.DEFINE_boolean("input_dropout",False,"Do dropout on input layer")
  configs.DEFINE_boolean("hidden_dropout",False,"Do dropout on hidden layers")
  configs.DEFINE_boolean("rnn_dropout",False,"Do dropout on recurrent connections")
  configs.DEFINE_boolean("skip_connections",False,"Have direct connections between input and output in MLP")
  configs.DEFINE_boolean("use_cache",True,"Load data for logreg from cache (vs processing from batch generator)")
  configs.DEFINE_boolean("pretty_print_preds",False,"Print predictions in tabular format with inputs, targets, and keys")
  configs.DEFINE_boolean("scale_targets",True,"")
  configs.DEFINE_string("data_scaler",None,'sklearn scaling algorithm or None if no scaling')
  configs.DEFINE_string("optimizer", 'GradientDescentOptimizer', 'Any tensorflow optimizer in tf.train')
  configs.DEFINE_string("optimizer_params",None, 'Additional optimizer params such as momentum')
  configs.DEFINE_float("learning_rate",0.6,"")
  configs.DEFINE_float("lr_decay",0.9, "Learning rate decay")
  configs.DEFINE_float("validation_size",0.0,"Size of validation set as %")
  configs.DEFINE_float("passes",1.0,"Passes through day per epoch")
  configs.DEFINE_float("target_lambda",0.5,"How much to weight last step vs. all steps in loss")
  configs.DEFINE_float("rnn_lambda",0.5,"How much to weight last step vs. all steps in loss")
  configs.DEFINE_integer("max_epoch",0,"Stop after max_epochs")
  configs.DEFINE_integer("early_stop",None,"Early stop parameter")
  configs.DEFINE_integer("seed",None,"Seed for deterministic training")
  configs.DEFINE_integer("cache_id",None,"A unique experiment key for traking a cahce")

  c = configs.ConfigValues()

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


def main():

    config = get_configs()
    train_path = model_utils.get_data_path(config.data_dir,config.datafile)

    print("Loading training data ...")

    config.batch_size = 1
    batches = BatchGenerator(train_path,config).valid_batches()

    params = batches.get_scaling_params('StandardScaler')

    #print(params['scale'])
    #print(params['center'])

    col_names = batches.feature_names
    df = pd.DataFrame(columns=col_names)
    
    gvkeys = list()
    dates  = list()
    steps  = list()

    print("Num batches sampled: %d"%batches.num_batches)
    for j in range(batches.num_batches):
        b = batches.next_batch()
        for i in range(config.num_unrollings):
            gvkeys.append( b.attribs[0][0] )
            dates.append( b.attribs[0][1] )
            steps.append( i )
            # x = (b.inputs[i][0] - params['center']) / params['scale']
            x = b.inputs[i][0] 
            n = len(df.index)
            df.loc[n] = x
        if (j % 1000)==0:
            print(".",end='')
            sys.stdout.flush()
    print()
            
    df = pd.concat( [df, pd.DataFrame( {'gvkey' : gvkeys, 'date': dates, 'step' : steps } )], axis=1 )
    
    for feature in col_names:
        mean = np.mean( df[feature] )
        std = np.std( df[feature] )
        print("%s %.4f %.4f"%(feature,mean,std))

    print('--------------------------------')

    for feature in col_names:
        print("%s:"%feature)
        st = df.sort_values(feature)
        for i in range(10):
            min_el = st.iloc[i,:]
            max_el = st.iloc[-i-1,:]
            print("%s %s %d %.4f"%
                      (min_el['gvkey'],min_el['date'],min_el['step'],min_el[feature]),end=' ')
            print("%s %s $d %.4f"%
                      (max_el['gvkey'],max_el['date'],max_el['step'],max_el[feature]))
        print('--------------------------------')
    
def other():
    
    df = train_data.dataframe

    dff = df[train_data.feature_names]

    df_norm = (dff - params['center']) / params['scale']

    df_norm = pd.concat((df.gvkey,df.date,df_norm),1)

    print(df_norm.head())

    for feature in train_data.feature_names:
        mean = np.mean( df_norm[feature] )
        std = np.std( df_norm[feature] )
        print("%s %.4f %.4f"%(feature,mean,std))

    print('--------------------------------')

    for feature in train_data.feature_names:
        print("%s:"%feature)
        st = df_norm.sort_values(feature)
        for i in range(5):
            min_el = st.iloc[i,:]
            max_el = st.iloc[-i-1,:]
            print("%s %s %.4f"%
                      (min_el['gvkey'],min_el['date'],min_el[feature]),end=' ')
            print("%s %s %.4f"%
                      (max_el['gvkey'],max_el['date'],max_el[feature]))
        print('--------------------------------')
        
if __name__ == "__main__":
    main()


    

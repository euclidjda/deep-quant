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
from batch_generator import BatchGenerator

def main():

    configs.DEFINE_integer("num_unrollings",5,"Number of unrolling steps")
    configs.DEFINE_integer("stride",12,"How many steps to skip per unrolling")
    configs.DEFINE_integer("batch_size",1,"Size of each batch")
    configs.DEFINE_string("datafile",'source-ml-data-100M-train.dat',"a datafile name.")
    configs.DEFINE_string("data_dir",'datasets',"The data directory")
    configs.DEFINE_float("validation_size",0.10,"Size of validation set as %")
    configs.DEFINE_string("scale_field", 'mrkcap',"Feature to scale inputs by")
    configs.DEFINE_string("key_field", 'gvkey',"Key column name header in datafile")
    configs.DEFINE_string("target_field", 'target',"Target column name header in datafile")
    configs.DEFINE_string("first_feature_field", 'saleq_ttm',"First feature")
    configs.DEFINE_integer("num_inputs",16,"Number of inputs")
    configs.DEFINE_integer("end_date",299912,"Last date to train on as YYYYMM")
    configs.DEFINE_integer("seed",1024,"Seed for deterministic training")

    config = configs.ConfigValues()

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

    print("Num batches sampled: %d"%batches.num_batches)
    for j in range(batches.num_batches):
        b = batches.next_batch()
        for i in range(config.num_unrollings):
            gvkeys.append( b.attribs[i][0][0] )
            dates.append( b.attribs[i][0][1] )
            # x = (b.inputs[i][0] - params['center']) / params['scale']
            x = b.inputs[i][0] 
            n = len(df.index)
            df.loc[n] = x
        if (j % 1000)==0:
            print(".",end='')
            sys.stdout.flush()
    print()
            
    df = pd.concat( [df, pd.DataFrame( {'gvkey' : gvkeys, 'date': dates } )], axis=1 )
    
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
            print("%s %s %.4f"%
                      (min_el['gvkey'],min_el['date'],min_el[feature]),end=' ')
            print("%s %s %.4f"%
                      (max_el['gvkey'],max_el['date'],max_el[feature]))
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


    

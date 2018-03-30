#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''

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

from utils import model_utils
import deep_quant
import configs as configs
from batch_generator import BatchGenerator

def main():

    config = deep_quant.get_configs()
    train_path = utils.data_utils.get_data_path(config.data_dir,config.datafile)

    print("Loading training data ...")

    config.batch_size = 1
    batches = BatchGenerator(train_path,config)
    # batches.cache(verbose=True)
    # batches.shuffle()

    params = batches.get_scaling_params('StandardScaler')

    print(params['scale'])
    print(params['center'])

    col_names = batches.feature_names
    df = pd.DataFrame(columns=col_names)
    
    gvkeys = list()
    dates  = list()
    steps  = list()

    print("Num batches sampled: %d"%batches.num_batches)
    for j in range(batches.num_batches):
    # for j in range(5000):
        b = batches.next_batch()
        seq_len = b.seq_lengths[0]
        idx = seq_len-1
        for i in range(seq_len):
            gvkeys.append( b.attribs[idx][0][0] )
            dates.append( b.attribs[idx][0][1] )
            steps.append( i )
            x = (b.inputs[i][0] - params['center']) / params['scale']
            # x = b.inputs[i][0] 
            n = len(df.index)
            df.loc[n] = x
        if (j % 1000)==0:
            print(".",end='')
            sys.stdout.flush()
    print()
            
    df = pd.concat( [pd.DataFrame( {'gvkey' : gvkeys, 'date': dates, 'step' : steps } ), df], axis=1 )

    # write to outfile
    df.to_csv(config.mse_outfile,sep=' ',float_format="%.4f")
    
    # print feature charateristics
    for feature in col_names:
        mean = np.mean( df[feature] )
        std = np.std( df[feature] )
        print("%s %.4f %.4f"%(feature,mean,std))

    print('--------------------------------')
    
    # print min and max values
    for feature in col_names:
        print("%s:"%feature)
        st = df.sort_values(feature)
        rt = df.sort_values(feature, ascending=False)
        for i in range(5):
            min_el = st.iloc[i,:]
            max_el = rt.iloc[i,:]
            #print(min_el)
            #print(max_el)
            print("%s %s %s %s"%
                      (min_el['gvkey'],min_el['date'],min_el['step'],min_el[feature]),end=' ')
            print("%s %s %s %s"%
                      (max_el['gvkey'],max_el['date'],max_el['step'],max_el[feature]))
        print('--------------------------------')

    
if __name__ == "__main__":
    main()


    

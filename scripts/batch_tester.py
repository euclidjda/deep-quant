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

import model_utils
import configs as configs
from batch_generator import BatchGenerator


def main():

    configs.DEFINE_integer("num_unrollings",5,"Number of unrolling steps")
    configs.DEFINE_integer("predict_steps",1,"Average future preds over this many steps")
    configs.DEFINE_integer("stride",12,"How many steps to skip per unrolling")
    configs.DEFINE_integer("batch_size",1,"Size of each batch")
    configs.DEFINE_string("datafile",'source-ml-data-dev.dat',"a datafile name.")
    configs.DEFINE_string("data_dir",'datasets',"The data directory")
    configs.DEFINE_float("validation_size",0.3,"Size of validation set as %")
    configs.DEFINE_string("key_field", 'gvkey',"Key column name header in datafile")
    configs.DEFINE_string("scale_field", 'mrkcap',"Feature to scale inputs by")
    configs.DEFINE_string("target_field", 'target',"Target column name header in datafile")
    configs.DEFINE_string("active_field", 'active',"Target column name header in datafile")
    configs.DEFINE_string("first_feature_field", 'saleq_ttm',"First feature")
    configs.DEFINE_integer("num_inputs",1,"Number of inputs")
    configs.DEFINE_integer("end_date",210001,"Last date to train on as YYYYMM")
    configs.DEFINE_integer("seed",1024,"Seed for deterministic training")

    config = configs.ConfigValues()

    train_path = model_utils.get_data_path(config.data_dir,config.datafile)

    print("Loading training data ...")

    batches = BatchGenerator(train_path,config,require_targets=True)

    print("Num batches %d"%batches.num_batches)

    #train_data = batches.train_batches()

    #print("Steps per epoch: %d"%train_data.num_batches)

    while(1):
        start_time = time.time()
        for step in range(batches.num_batches):
            batch = batches.next_batch()

            key     = batch.attribs[-2][0][0]
            ndate   = batch.attribs[0][0][1]
            pdate   = batch.attribs[-2][0][1]
            edate   = batch.attribs[-1][0][1]
            inputs  = batch.inputs[-1][0]
            targets = batch.targets[-1][0]
            scale   = batch.seq_scales[0]

            print("%s %s %s %s %d %d %d"%
                      (key,ndate,pdate,edate,len(batch.attribs),len(batch.inputs),len(batch.targets)))
            sys.stdout.flush()

        speed = time.time() - start_time
        print("\nEpoch time is %0f seconds" % speed)
        sys.stdout.flush()
        exit()

if __name__ == "__main__":
    main()

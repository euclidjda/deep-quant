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

from utils import data_utils
import configs as configs
import deep_quant
from batch_generator import BatchGenerator
from profilehooks import profile

@profile
def batching(batches):
    for i in range(10):
        start_time = time.time()
        for _ in range(100):
            batch = batches.next_batch()

            #key     = batch.attribs[-2][0][0]
            #ndate   = batch.attribs[0][0][1]
            #pdate   = batch.attribs[-2][0][1]
            #edate   = batch.attribs[-1][0][1]
            #inputs  = batch.inputs[-1][0]
            #targets = batch.targets[-1][0]

            #print("%s %s %s %s %d %d %d"%
            #          (key,ndate,pdate,edate,len(batch.attribs),len(batch.inputs),len(batch.targets)))
            #sys.stdout.flush()

        speed = time.time() - start_time
        print("Epoch time is %0f seconds" % speed)
        sys.stdout.flush()

def main():

    config = deep_quant.get_configs()

    data_path = data_utils.get_data_path(config.data_dir,config.datafile)

    print("Loading data ..."); sys.stdout.flush()

    batches = BatchGenerator(data_path,config,require_targets=True)

    print("Num batches %d"%batches.num_batches); sys.stdout.flush()
    
    batching(batches)



if __name__ == "__main__":
    main()

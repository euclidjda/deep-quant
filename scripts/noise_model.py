# Copyright 2016 Euclidean Technologies Management LLC  All Rights Reserved.
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

import os
import time
import sys
import random

import numpy as np
import pandas as pd
import math
import copy

class NoiseModel(object):
    def __init__(self, seed=None, scaling_params=None, degree=0.00):
        self._scale = scaling_params['scale']
        self._degree = degree
        if seed is not None:
            np.random.seed(seed)

    def add_noise(self,batch):
        batch = copy.deepcopy(batch)
        inputs = batch.inputs
        input_scales = np.tile(self._scale,(batch.size,1))
        num_inputs = inputs[0].shape[1]
        for i in range(len(inputs)):
            input_noise = np.random.normal(loc=0.0,
                                           scale=self._degree,
                                           size=(batch.size,num_inputs))
            batch.inputs[i] += input_scales * input_noise
        return batch

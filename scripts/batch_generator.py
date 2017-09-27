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
import numpy as np
import pandas as pd
import random
import sklearn.preprocessing

_NUM_CLASSES = 2

class BatchGenerator(object):
    """
    BatchGenerator object takes a data file are returns an object with
    a next_batch() function. The next_batch() function yields a batch of data
    sequences from the datafile whose shape is specified by config.batch_size
    and config.num_unrollings.
    """
    def __init__(self, filename, config, validation=True, data=None):
        """
        Init a BatchGenerator
        """
        self._key_name = key_name = config.key_field
        self._target_name = target_name = config.target_field
        self._first_feature_name = first_feature_name = config.first_feature_field
        self._num_inputs = config.num_inputs
        self._num_unrollings = num_unrollings = config.num_unrollings
        self._stride = config.stride
        self._batch_size = batch_size = config.batch_size

        assert( self._stride >= 1 )

        self._rnn_loss_weight = None
        if hasattr(config,'rnn_loss_weight'):
            self._rnn_loss_weight = config.rnn_loss_weight
        
        self._config = config # save this around for train_batches() method
        
        if data is None:
            if not os.path.isfile(filename):
                raise RuntimeError("The data file %s does not exists" % filename)
            data = pd.read_csv(filename,sep=' ', dtype={ self._key_name : str } )
            if config.end_date is not None:
                data = data.drop(data[data['date'] > config.end_date].index)

        self._end_date = data['date'].max()
        self._start_date = data['date'].min()
        self._feature_start_idx = list(data.columns.values).index(first_feature_name)
        self._key_idx = list(data.columns.values).index(key_name)
        self._target_idx = list(data.columns.values).index(target_name)
        self._date_idx = list(data.columns.values).index('date')
        self._feature_names = list(data.columns.values)[self._feature_start_idx:]
        assert(self._feature_start_idx>=0)

        # This assert ensures that no x features are the yval
        assert(list(data.columns.values).index(target_name)
                   < self._feature_start_idx)
        self._data = data
        self._data_len = len(data)
        self._validation_set = dict()

        if validation is True:
            if config.seed is not None:
                print("setting random seed to "+str(config.seed))
                random.seed( config.seed )
            # get number of keys
            keys = list(set(data[key_name]))
            keys.sort()
            sample_size = int( config.validation_size * len(keys) )
            sample = random.sample(keys, sample_size)
            self._validation_set = dict(zip(sample,[1]*sample_size))
            print("Num training entities: %d"%(len(keys)-sample_size))
            print("Num validation entities: %d"%sample_size)

        # Setup indexes into the sequences
        self._seq_length = min_seq_length = self._stride * (num_unrollings+1)
        self._indices = list()
        last_key = ""
        cur_length = 1
        for i in range(self._data_len):
            key = data.iat[i,self._key_idx]
            if (key != last_key):
                cur_length = 1
            if (cur_length >= min_seq_length):
                # TODO: HERE WE COULD OVER-SAMPLE BASED ON
                # DATE TO MORE HEAVILY WEIGHT MORE RECENT
                self._indices.append(i-min_seq_length+1)
            cur_length += 1
            last_key = key

        # Create a cursor of equally spaced indices into the dataset. Each index
        # in the cursor points to one sequence in a batch and is used to keep
        # track of where we are in the dataset.
        batch_size = self._batch_size
        num_batches = len(self._indices) // batch_size
        self._index_cursor = [ offset * num_batches for offset in range(batch_size) ]
        self._init_index_cursor = self._index_cursor[:]
        self._num_batches = num_batches
        self._batch_cache = [None]*num_batches
        self._batch_cursor = 0
        
    def _next_step(self, step):
        """
        Get next step in current batch.
        """
        x = np.zeros(shape=(self._batch_size, self._num_inputs), dtype=np.float)
        attr = list()
        data = self._data
        features_idx = self._feature_start_idx
        num_inputs = self._num_inputs
        key_idx = self._key_idx
        date_idx = self._date_idx
        stride = self._stride
        for b in range(self._batch_size):
            cursor = self._index_cursor[b]
            start_idx = self._indices[cursor]
            end_idx = start_idx + self._seq_length - 1
            idx = start_idx + (step*stride)
            assert(idx <= end_idx)
            x[b,:] = data.iloc[idx,features_idx:features_idx+num_inputs].as_matrix()
            date = data.iat[idx,date_idx]
            key = data.iat[idx,key_idx]
            attr.append((key,date))

        return x, attr

    def _next_batch(self):
        """Generate the next batch of sequences from the data.
        Returns:
          A batch of type Batch (see class def below)
        """
        batch_data = list()
        attributes = list()
        for i in range(self._num_unrollings+1):
            data, attr = self._next_step(i)
            batch_data.append(data)
            attributes.append(attr)

        inputs  = batch_data[0:-1]
        targets = batch_data[1:]
        assert(len(inputs)==len(targets))
        
        #############################################################################
        #   Set cursor for next batch
        #############################################################################
        batch_size = self._batch_size
        num_idxs = len(self._indices)
        self._index_cursor = [ (self._index_cursor[b]+1)%num_idxs for b in range(batch_size) ]

        return Batch(inputs, targets, attributes)

    def next_batch(self):
        b = None
        if self._batch_cache[self._batch_cursor] is not None:
            b = self._batch_cache[self._batch_cursor]
        else:
            b = self._next_batch()
            self._batch_cache[self._batch_cursor] = b
        self._batch_cursor = (self._batch_cursor+1) % (self._num_batches)

        return b

    def get_scaling_params(self,scaler_class):

        scaler = None
        
        if hasattr(sklearn.preprocessing,scaler_class):
            scaler = getattr(sklearn.preprocessing,scaler_class)()
        else:
            raise RuntimeError("Unknown scaler = %s"%scaler_class)

        start_idx = self._feature_start_idx
        end_idx   = start_idx+self._num_inputs
        fdata = self._data.iloc[:,start_idx:end_idx]
        scaler.fit(fdata)
        params = dict()
        params['center'] = scaler.center_
        params['scale'] = scaler.scale_
        
        return params
        
    def train_batches(self):
        valid_keys = list(self._validation_set.keys())
        indexes = self._data[self._key_name].isin(valid_keys)
        train_data = self._data[~indexes]
        return BatchGenerator("",self._config,validation=False,
                                  data=train_data)

    def valid_batches(self):
        valid_keys = list(self._validation_set.keys())
        indexes = self._data[self._key_name].isin(valid_keys)
        valid_data = self._data[indexes]
        return BatchGenerator("",self._config,validation=False,
                                  data=valid_data)

    def shuffle(self):
        random.shuffle(self._batch_cache)
        self._batch_cusror = 0
         
    def rewind(self):
        self._batch_cusror = 0

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def num_unrollings(self):
        return self._num_unrollings
    
class Batch(object):
    """
    """

    def __init__(self,inputs,targets, attribs):
        self._inputs = inputs
        self._targets = targets
        self._attribs = attribs

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def attribs(self):
        return self._attribs


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

_MIN_SEQ_SCALE = 1.0

class BatchGenerator(object):
    """
    BatchGenerator object takes a data file are returns an object with
    a next_batch() function. The next_batch() function yields a batch of data
    sequences from the datafile whose shape is specified by config.batch_size
    and config.num_unrollings.
    """
    def __init__(self, filename, config, validation=True, require_targets=True,
                     data=None, verbose=True):
        """
        Init a BatchGenerator
        """
        self._key_name = key_name = config.key_field
        self._target_name = target_name = config.target_field
        self._scaling_feature = config.scale_field
        self._num_unrollings = num_unrollings = config.num_unrollings
        self._predict_steps = config.predict_steps
        self._stride = config.stride
        self._batch_size = batch_size = config.batch_size
        
        assert( self._stride >= 1 )

        if data is None:
            if not os.path.isfile(filename):
                raise RuntimeError("The data file %s does not exists" % filename)
            data = pd.read_csv(filename,sep=' ', dtype={ self._key_name : str } )
            if config.end_date is not None:
                data = data.drop(data[data['date'] > config.end_date].index)

        assert( config.feature_fields.find('-') < len(config.feature_fields)-1 )
        assert( config.feature_fields.find('-') > 0 )
        (first_feature_name,last_feature_name) = config.feature_fields.split('-')
        self._feature_start_idx = list(data.columns.values).index(first_feature_name)
        self._feature_end_idx = list(data.columns.values).index(last_feature_name)
        assert(self._feature_start_idx>=0)
        assert( self._feature_start_idx <= self._feature_end_idx )
        
        self._feature_names = list(data.columns.values)[self._feature_start_idx:self._feature_end_idx+1]
        config.num_features = self._num_features = self._feature_end_idx - self._feature_start_idx + 1
        assert( self._num_features == len(self._feature_names) )

        self._key_idx = list(data.columns.values).index(key_name)
        self._target_idx = list(data.columns.values).index(target_name)
        self._active_idx = list(data.columns.values).index(config.active_field)
        self._date_idx = list(data.columns.values).index('date')

        self._config = config # save this around for train_batches() method
        
        # This assert ensures that no x features are the yval
        assert(list(data.columns.values).index(target_name)
                   < self._feature_start_idx)
        self._data = data
        self._data_len = len(data)

        # Setup the validation data
        self._validation_set = dict()
        if validation is True:
            if config.seed is not None:
                if verbose is True: print("setting random seed to "+str(config.seed))
                random.seed( config.seed )
            # get number of keys
            keys = list(set(data[key_name]))
            keys.sort()
            sample_size = int( config.validation_size * len(keys) )
            sample = random.sample(keys, sample_size)
            self._validation_set = dict(zip(sample,[1]*sample_size))
            if verbose is True:
                print("Num training entities: %d"%(len(keys)-sample_size))
                print("Num validation entities: %d"%sample_size)

        # Setup indexes into the sequences
        min_seq_length = self._stride * (num_unrollings-1) + 1
        steps = self._predict_steps*self._stride
        self._indices = list()
        last_key = ""
        cur_length = 1
        for i in range(self._data_len):
            # get active value
            key = data.iat[i,self._key_idx]
            pred_key = data.iat[i+steps, self._key_idx] if i+steps < len(data) else ""
            active = True if int(data.iat[i,self._active_idx]) else False
            if (key != last_key):
                cur_length = 1
            if ( (cur_length >= min_seq_length) and (active is True) ):
                # If targets are not required, we don't need the future
                # sequences to be there, otherwise we do
                if (not require_targets) or (key == pred_key):
                    self._indices.append(i-min_seq_length+1)
            cur_length += 1
            last_key = key

        if verbose is True:
            print("Number of batch indices: %d"%(len(self._indices)))
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
        x = np.zeros(shape=(self._batch_size, self._num_features), dtype=np.float)
        y = np.zeros(shape=(self._batch_size, self._num_features), dtype=np.float)
        
        attr = list()
        data = self._data
        key_idx = self._key_idx
        date_idx = self._date_idx
        stride = self._stride
        for b in range(self._batch_size):
            cursor = self._index_cursor[b]
            start_idx = self._indices[cursor]
            idx = start_idx + (step*stride)
            date = None
            key = None
            assert( idx < self._data_len )
            date = data.iat[idx,date_idx]
            key = data.iat[idx,key_idx]
            x[b,:] = self._get_feature_vector(start_idx,step)
            y[b,:] = self._get_feature_vector(start_idx,step+1)
            attr.append((key,date))
        return x, y, attr

    def _next_batch(self):
        """Generate the next batch of sequences from the data.
        Returns:
          A batch of type Batch (see class def below)
        """
        scales = self._get_sequence_scales( )
        
        inputs = list()
        targets = list()
        attribs = None
        for i in range(self._num_unrollings):
            x, y, attr = self._next_step(i)
            inputs.append(x)
            targets.append(y)
            attribs = attr

        assert(len(inputs)==len(targets))
        
        #############################################################################
        #   Set cursor for next batch
        #############################################################################
        batch_size = self._batch_size
        num_idxs = len(self._indices)
        self._index_cursor = [ (self._index_cursor[b]+1)%num_idxs for b in range(batch_size) ]

        return Batch(inputs, targets, attribs, scales)

    def next_batch(self):
        b = None
        if self._batch_cache[self._batch_cursor] is not None:
            b = self._batch_cache[self._batch_cursor]
        else:
            b = self._next_batch()
            self._batch_cache[self._batch_cursor] = b
        self._batch_cursor = (self._batch_cursor+1) % (self._num_batches)

        return b

    def _get_scale(self,start_idx):
        scale_idx = start_idx + (self._num_unrollings-1)*self._stride
        s = max(self._data.iloc[scale_idx][self._scaling_feature],_MIN_SEQ_SCALE)
        return s
        
    def _get_sequence_scales(self):
        scales = list()
        
        for b in range(self._batch_size):
            cursor = self._index_cursor[b]
            start_idx = self._indices[cursor]
            s = self._get_scale(start_idx)
            scales.append(s)
        return np.array( scales )
           
    def _get_feature_vector(self,start_idx,cur_step):
        features_idx = self._feature_start_idx
        stride = self._stride
        data = self._data
        cur_idx = start_idx+cur_step*stride
        if cur_idx < self._data_len:
            s = self._get_scale(start_idx)
            x = data.iloc[cur_idx,self._feature_start_idx:self._feature_end_idx+1].as_matrix()
            y = np.divide(x,s)
            y_abs = np.absolute(y).astype(float)
            return np.multiply(np.sign(y),np.log1p(y_abs))
        else:
            return np.zeros(shape=[self._num_features])
        
    def get_scaling_params(self,scaler_class):
        features_idx = self._feature_start_idx
        num_features = self._num_features
        stride = self._stride
        data = self._data
        
        sample = list()
        for i in self._indices:
            step = np.random.randint(self._num_unrollings)
            sample.append(self._get_feature_vector(i,step))

        scaler = None
        
        if hasattr(sklearn.preprocessing,scaler_class):
            scaler = getattr(sklearn.preprocessing,scaler_class)()
        else:
            raise RuntimeError("Unknown scaler = %s"%scaler_class)

        scaler.fit(sample)
        params = dict()
        params['center'] = scaler.center_ if hasattr(scaler,'center_') else scaler.mean_
        params['scale'] = scaler.scale_

        return params
            
    def train_batches(self):
        valid_keys = list(self._validation_set.keys())
        indexes = self._data[self._key_name].isin(valid_keys)
        train_data = self._data[~indexes]
        return BatchGenerator("", self._config, validation=False,
                                  data=train_data)

    def valid_batches(self):
        valid_keys = list(self._validation_set.keys())
        indexes = self._data[self._key_name].isin(valid_keys)
        valid_data = self._data[indexes]
        return BatchGenerator("", self._config, validation=False,
                                  data=valid_data)

    def shuffle(self):
        # We cannot shuffle until the entire dataset is cached
        if (self._batch_cache[-1] is not None):
            random.shuffle(self._batch_cache)
            self._batch_cusror = 0
         
    def rewind(self):
        self._batch_cusror = 0

    @property
    def feature_names(self):
        return self._feature_names
        
    @property
    def dataframe(self):
        return self._data

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def num_unrollings(self):
        return self._num_unrollings

    @property
    def num_features(self):
        return self._num_features
    
class Batch(object):
    """
    """

    def __init__(self, inputs, targets, attribs, seq_scales):
        self._inputs = inputs
        self._targets = targets
        self._attribs = attribs
        self._seq_scales = seq_scales

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def attribs(self):
        return self._attribs

    @property
    def seq_scales(self):
        return self._seq_scales
    

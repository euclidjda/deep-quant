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

_NUM_CLASSES = 2

class BatchGenerator(object):
    """
    BatchGenerator object takes a data file are returns an object with
    a next_batch() function. The next_batch() function yields a batch of data
    sequences from the datafile whose shape is specified by config.batch_size
    and config.num_unrollings.
    """
    def __init__(self, filename, config, randomly_sample=False, validation=True, data=None):
        """
        Init a BatchGenerator
        """
        self._key_name = key_name = config.key_field
        self._target_name = target_name = config.target_field
        self._randomly_sample = randomly_sample
        self._num_classes = config.num_outputs
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
        self._feature_start_idx = list(data.columns.values).index(target_name)+1
        self._key_idx = list(data.columns.values).index(key_name)
        self._target_idx = list(data.columns.values).index(target_name)
        self._date_idx = list(data.columns.values).index('date')
        self._feature_names = list(data.columns.values)[self._target_idx+1:]
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
            #print("\n".join(sample))

        # Setup indexes into the sequences
        self._seq_length = min_seq_length = self._stride * num_unrollings
        self._indices = list()
        last_key = ""
        cur_length = 1
        for i in range(self._data_len):
            key = data.iat[i,self._key_idx]
            if (key != last_key):
                cur_length = 1
            if (cur_length >= min_seq_length):
                #
                # TODO: HERE WE COULD OVER-SAMPLE BASED ON
                # DATE TO MORE HEAVILY WEIGHT MORE RECENT
                # 
                self._indices.append(i-min_seq_length+1)
                #self._end_indices.append(i)
                #self._indices.append(i-seq_length+1)
                # print("%d %d %d"%(seq_length,i,i-seq_length+1))
            cur_length += 1
            last_key = key

        # Create a cursor of equally spaced indices into the dataset. Each index
        # in the cursor points to one sequence in a batch and is used to keep
        # track of where we are in the dataset.
        batch_size = self._batch_size
        num_batches = len(self._indices) // batch_size
        self._cursor = [ offset * num_batches for offset in range(batch_size) ]
        self._init_cursor = self._cursor[:]
        self._num_batches = num_batches

    def _target_to_class_idx(self, target_val):
        n = self._num_classes
        assert( n > 0 )
        class_idx = 0
        if target_val == 1.0:
            class_idx = n-1
        else:
            class_idx = int(target_val*n)
        return class_idx

    def _next_step(self, step, seq_lengths):
        """
        Get next step in current batch.
        """
        x = np.zeros(shape=(self._batch_size, self._num_inputs), dtype=np.float)
        y = np.zeros(shape=(self._batch_size, self._num_classes), dtype=np.float)
        train_mask = np.zeros(shape=(self._batch_size), dtype=np.float)
        valid_mask = np.zeros(shape=(self._batch_size), dtype=np.float)
        attr = list()
        data = self._data
        features_idx = self._feature_start_idx
        num_inputs = self._num_inputs
        key_idx = self._key_idx
        target_idx = self._target_idx
        date_idx = self._date_idx
        stride = self._stride
        for b in range(self._batch_size):
            cursor = self._cursor[b]
            start_idx = self._indices[cursor]
            end_idx = start_idx + self._seq_length - 1
            seq_lengths[b] = int((end_idx-start_idx+1)//stride)
            assert(seq_lengths[b]>0)
            idx = start_idx + (step*stride)
            assert(idx <= end_idx)
            x[b,:] = data.iloc[idx,features_idx:features_idx+num_inputs].as_matrix()
            val = data.iat[idx,target_idx] 
            class_idx = self._target_to_class_idx( val )
            y[b,class_idx] = 1.0
            date = data.iat[idx,date_idx]
            key = data.iat[idx,key_idx]
            attr.append((key,date))

        return x, y, train_mask, valid_mask, attr

    def next_batch(self):
        """Generate the next batch of sequences from the data.
        Returns:
          A batch of type Batch (see class def below)
        """
        seq_lengths = np.full(self._batch_size, self._num_unrollings, dtype=int) 
        x_batch = list()
        y_batch = list()
        train_mask = list()
        valid_mask = list()
        attribs = list()
        for i in range(self._num_unrollings):
            x, y, tw, vw, attr = self._next_step(i, seq_lengths)
            x_batch.append(x)
            y_batch.append(y)
            train_mask.append(tw)
            valid_mask.append(vw)
            attribs.append(attr)

        #############################################################################
        #   Set cursor for next batch
        #############################################################################
        batch_size = self._batch_size
        if self._randomly_sample is True:
            self._cursor = random.sample(range(len(self._indices)),batch_size)
        else:
            num_idxs = len(self._indices)
            self._cursor = [ (self._cursor[b]+1)%num_idxs for b in range(batch_size) ]

        return Batch(x_batch, y_batch, seq_lengths,
                         train_mask, valid_mask, attribs )

    def train_batches(self):
        valid_keys = list(self._validation_set.keys())
        indexes = self._data[self._key_name].isin(valid_keys)
        train_data = self._data[~indexes]
        return BatchGenerator("",self._config,validation=False,
                                  randomly_sample=True,
                                  data=train_data)

    def valid_batches(self):
        valid_keys = list(self._validation_set.keys())
        indexes = self._data[self._key_name].isin(valid_keys)
        valid_data = self._data[indexes]
        return BatchGenerator("",self._config,validation=False,
                                  randomly_sample=False,
                                  data=valid_data)

    def rewind(self):
        self._cursor = self._init_cursor[:]

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def num_unrollings(self):
        return self._num_unrollings
    
class Batch(object):
    """
    """

    def __init__(self,inputs,targets,seq_lengths,
                     train_mask,valid_mask, attribs):
        self._inputs = inputs
        self._targets = targets
        self._seq_lengths = seq_lengths
        self._train_mask = train_mask
        self._valid_mask = valid_mask
        self._attribs = attribs

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def seq_lengths(self):
        return self._seq_lengths

    @property
    def train_mask(self):
        return self._train_mask

    @property
    def valid_mask(self):
        return self._valid_mask

    @property
    def attribs(self):
        return self._attribs


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
import pickle
import hashlib

import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

_MIN_SEQ_NORM = 10.0
DEEP_QUANT_ROOT = os.environ['DEEP_QUANT_ROOT']
DATASETS_PATH = os.path.join(DEEP_QUANT_ROOT, 'datasets')

class BatchGenerator(object):
    """
    BatchGenerator takes a data file and returns an object with a next_batch()
    function. The next_batch() function yields a batch of data sequences from
    the datafile whose shape is specified by config.batch_size and
    config.max_unrollings.
    """
    def __init__(self, filename, config, validation=True, require_targets=True,
                     data=None, verbose=True, is_training_only=False):
        """
        Init a BatchGenerator.
        Data is loaded as a Pandas DataFrame and stored in the _data attribute.
        `
        Sequences of length config.max_unrollings are identified; the start and
        end indices of these sequences are stored in the _start_indices and
        _end_indices attributes of our BatchGenerator object, respectively. 
        (These are both lists of equal length that indicate the indices of the
        start and of the end of the sequence, within the _data DataFrame.)
        Indices are set up to access config.batch_size of these sequences as a
        batch; these are set up in such way that these sequences are "far apart"
        from each other in _data (they're num_batches rows away from each other,
        more specifically). This way it's unlikely that more than one sequence
        corresponds to the same company in a batch.
        """
        self._scaling_feature = config.scale_field
        self._max_unrollings = config.max_unrollings
        self._min_unrollings = config.min_unrollings
        self._stride = config.stride
        self._forecast_n = config.forecast_n
        self._batch_size = config.batch_size
        self._scaling_params = None
        self._log_squasher = config.log_squasher
        self._start_date = config.start_date
        self._end_date = config.end_date
        self._is_training_only = is_training_only
        self._ts_smoother = config.ts_smoother
        self._backfill = config.backfill

        assert self._stride >= 1

        self._init_data(filename, config, validation, data, verbose)
        self._init_batch_cursor(config, require_targets, verbose)
        self._config = config # save this around for train_batches() method

    def _init_data(self, filename, config, validation=True, data=None, 
                   verbose=True):
        """
        Loads data as Pandas DataFrame, saves it as ._data attribute, sets up a
        validation set if necessary. 
        (The latter consists of creating a dictionary to house the unique
        identifiers of the companies that are to be used for validation, and
        populating it with randomly-selected companies.)
        """
        # Load data
        if data is None:
            if not os.path.isfile(filename):
                raise RuntimeError("The data file %s does not exist" % filename)
            data = pd.read_csv(filename, sep=' ', 
                               dtype={config.key_field: str})
            if config.end_date is not None:
                data = data.drop(data[data[config.date_field] > config.end_date].index)

        # Define attributes
        self._keys = data[config.key_field].tolist()
        self._dates = data[config.date_field].tolist()
        self._data = data
        self._data_len = len(data)
        assert(self._data_len)

        print("Total number of records %d"%len(self._dates))

        # Setup data
        self._init_column_indices(config)
        self._init_validation_set(config, validation, verbose)

    def _init_batch_cursor(self, config, require_targets=True, verbose=True):
        """
        Sets up indexes into the sequences.
        First identifies start and end points of sequences (stored as
        _start_indices and _end_indices).
        Then sets up two cursors:
          (1) _index_cursor, which is a cursor of equally-spaced indices into
              the dataset. Here, each index points to a sequence (which can be
              determined fully using _data, _start_indices, and _end_indices).
              There will be config.batch_size indices in this list.
          (2) _batch_cursor, which keeps track of the batch that we're working
              with. (This is just an int that changes as we go through the
              dataset in batches.)
        Note that the number of batches is dictated by the number of sequences
        available and the user-defined size of each batch (as specified in
        config.batch_size). (The number of sequences available in turn depends
        on the length of those sequences, config.max_unrollings as well as the
        size of the dataset).
        Here, an attribute called _batch_cache is also created. This is a list
        of size num_batches that will house the contents of each batch once
        they're cached.
        Lastly, an attribute called _init_index_cursor is also created. This is
        simply a copy of _index_cursor in its original state, which will allow
        us to go back to the start if we need to once _index_cursor has changed.
        """
        data = self._data
        stride = self._stride
        min_steps = stride * (self._min_unrollings-1) + 1
        max_steps = stride * (self._max_unrollings-1) + 1
        forecast_n = self._forecast_n
        self._start_indices = list()
        self._end_indices = list()
        start_date = 100001
        if config.start_date is not None:
            start_date = config.start_date
        last_key = ""
        cur_length = 1
        
        # Identify start and end points of sequences 
        # TODO: roll up as function (to be housed within _init_batch_cursor)?
        for i in range(self._data_len):
            key = data.iat[i, self._key_idx]
            if i+forecast_n < len(data):
                pred_key = data.iat[i+forecast_n, self._key_idx]
            else:
                pred_key = ""
            active = True if int(data.iat[i,self._active_idx]) else False
            date = data.iat[i,self._date_idx]
            if key != last_key:
                cur_length = 1
            if ((cur_length >= min_steps)
                 and (active is True)
                 and (date >= start_date)):
                # If targets are not required, we don't need the future
                # sequences to be there, otherwise we do
                seq_len = min(cur_length-(cur_length-1)%stride, max_steps)
                if (not require_targets) or (key == pred_key):
                    self._start_indices.append(i-seq_len+1)
                    self._end_indices.append(i)
            cur_length += 1
            last_key = key

        if verbose is True:
            print("Number of batch indices: %d"%(len(self._start_indices)))

        # Create a cursor of equally spaced indices into the dataset. Each index
        # in the cursor points to one sequence in a batch and is used to keep
        # track of where we are in the dataset.
        batch_size = self._batch_size
        num_batches = len(self._start_indices) // batch_size
        self._index_cursor = [offset*num_batches for offset in range(batch_size)]
        self._init_index_cursor = self._index_cursor[:]
        self._num_batches = num_batches
        assert(num_batches > 0)
        self._batch_cache = [None]*num_batches
        self._batch_cursor = 0

    def _init_column_indices(self, config):
        """
        # TODO: rewrite this docstring
        Sets up column-index-related attributes and adds a few items to the
        config.
        Column-index-related attributes:
          * _feature_indices: A list housing the column numbers of the features,
                              where features are as specified by 
                              config.financial_fields.
          * _aux_indices: A list housing the column numbers of the auxilary
                          covariates, where these auxilary covariates are
                          specified by config.aux_fields.
          * _input_names: A list housing the names of the columns of the 
                            features _and_ of the auxilary covariates.
          * _num_inputs: The total number of covariates used as input (so both
                         those that are specified in config.financial_fields and
                         those that are specified in config.aux_fields).
          * _key_idx: The column index of what should be used as a unique
                      identifier of each company (the index of gvkey, for 
                      example).
          * _active_idx: The column index of the field that lets us know whether
                         a company was actively trading during a specific point
                         in time or not.
          * _date_idx: The column index of the date field.
        Items added to the config:
          * num_inputs: same as the _num_inputs attribute
          * num_ouptus: num_inputs minus the number of aux covariates.
          * target_idx: index of target variable within the list of features, if
                        target is specified by config.
        """
        assert config.financial_fields
        def get_colidxs_from_colnames(data, columns):
            """
            Returns indexes of columns of data that are included in columns.
            columns are seperated by commas and can include ranges. For example,
            f1-f5,f7,f9 whould be feature one through 5, and feature 7 and 9.
            """
            colidxs = []
            if columns is not None:
                colnames = list(data.columns.values)
                col_list = columns.split(',')
                for col in col_list:
                    col_range = col.split('-')
                    if len(col_range) == 1:
                        colidxs.append(list(colnames).index(col_range[0]))
                    elif len(col_range) == 2:
                        start_idx = list(colnames).index(col_range[0])
                        end_idx = list(colnames).index(col_range[1])
                        assert(start_idx >= 0)
                        assert(start_idx <= end_idx)
                        colidxs.extend(list(range(start_idx,end_idx+1)))
            return colidxs
        
        # Set up financials column indices and auxiliaries column indices
        self._fin_colidxs = get_colidxs_from_colnames(
            self._data, config.financial_fields)

        self._aux_colidxs = get_colidxs_from_colnames(
            self._data, config.aux_fields)

        all_colidxs = self._fin_colidxs + self._aux_colidxs

        # save feature names
        colnames = self._data.columns.values
        self._feature_names = colnames[all_colidxs]

        # store input vector indices to NOT scale
        dont_scale_colidxs = get_colidxs_from_colnames( self._data, config.dont_scale )
        dont_scale_colidxs = [i for i in dont_scale_colidxs if i in all_colidxs]
        self._dont_scale_input_idxs = [all_colidxs.index(i) for i in dont_scale_colidxs]

        # Set up other attributes
        colnames = list(colnames)
        self._key_idx = colnames.index(config.key_field)
        self._active_idx = colnames.index(config.active_field)
        self._date_idx = colnames.index(config.date_field)
        if config.scale_field == '__norm__':
            self._normalizer_idx = None
        else:
            self._normalizer_idx = colnames.index(config.scale_field)

        # Set up input-related attributes
        self._num_inputs = config.num_inputs = len(self._feature_names)

        # Set up target index
        idx = colnames.index(config.target_field)
        config.target_idx = idx - self._fin_colidxs[0]
        self._num_outputs = config.num_outputs = \
            self._num_inputs - len(self._aux_colidxs)
        self._price_target_idx = -1

        assert(config.target_idx >= 0)

        # Set up fin_inputs attribute and aux_inputs attribute
        self._fin_inputs  = self._data.iloc[:, self._fin_colidxs].as_matrix()
        self._aux_inputs  = self._data.iloc[:, self._aux_colidxs].as_matrix()

        if self._normalizer_idx is not None:
            self._normalizers = self._data.iloc[:, self._normalizer_idx].as_matrix()
        else:
            self._normalizers = np.linalg.norm(self._fin_inputs, axis=1)

    def _init_validation_set(self, config, validation, verbose=True):
        """
        Sets up validation set. Creates the _validation_set attribute, which is
        a dictionary housing the keys (unique identifier, such as gvkey) of
        the companies that should be used for validation as the dictionary's
        keys, and the number 1 as the dictionary's values.
        """
        # Setup the validation data
        self._validation_set = dict()
        if validation is True:
            if config.seed is not None:
                if verbose is True:
                    print("Setting random seed to "+str(config.seed))
                random.seed( config.seed )
                np.random.seed( config.seed )
            # get number of keys
            keys = sorted(set(self._data[config.key_field]))
            sample_size = int(config.validation_size * len(keys))
            sample = random.sample(keys, sample_size)
            self._validation_set = dict(zip(sample, [1]*sample_size))  

    def _get_normalizer(self, end_idx):
        val = max(self._normalizers[end_idx], _MIN_SEQ_NORM)
        return val

    def _get_batch_normalizers(self):
        """
        Returns an np.array housing the normalizers (scalers) by which the
        inputs of the current sequence should be scaled (this is specified by
        config.scale_field).
        """
        normalizers = list()
        for b in range(self._batch_size):
            cursor = self._index_cursor[b]
            end_idx = self._end_indices[cursor]
            s = self._get_normalizer(end_idx)
            normalizers.append(s)
        return np.array( normalizers )

    def _get_feature_vector(self,end_idx,cur_idx):
        if cur_idx < self._data_len:
            x = self._fin_inputs[cur_idx]
            # Time-series smoother
            if self._ts_smoother is True:
                if cur_idx < end_idx:
                    for i in range(cur_idx+1,end_idx+1):
                        x += self._fin_inputs[i]
                    x /= float(end_idx-cur_idx+1)
                elif (cur_idx > end_idx) and (self._is_training_only is True):
                    x += self._fin_inputs[end_idx]
                    x /= 2.0
            # Normalize feature vector by normalizing feature
            n = self._get_normalizer(end_idx)
            assert(n>0)
            y = np.divide(x,n)
            if self._log_squasher is True:
                y_abs = np.absolute(y).astype(float)
                y = np.multiply(np.sign(y),np.log1p(y_abs))
            return y
        else:
            return np.zeros(shape=[len(self._fin_colidxs)])

    def _get_aux_vector(self,cur_idx):
        if cur_idx < self._data_len:
            x = self._aux_inputs[cur_idx]
            return x
        else:
            return np.zeros(shape=[len(self._aux_colidxs)])

    def _next_step(self, step, seq_lengths):
        """
        Get next step in current batch.
        """
        x = np.zeros(shape=(self._batch_size, self._num_inputs), dtype=np.float)
        y = np.zeros(shape=(self._batch_size, self._num_outputs), dtype=np.float)

        attr = list()
        stride = self._stride
        forecast_n = self._forecast_n
        len1 = len(self._fin_colidxs)
        len2 = len(self._aux_colidxs)

        for b in range(self._batch_size):
            cursor = self._index_cursor[b]
            start_idx = self._start_indices[cursor]
            end_idx = self._end_indices[cursor]
            idx = start_idx
            if self._backfill is True:
                seq_length = ((end_idx-start_idx)//stride)+1
                diff = self._max_unrollings - seq_length
                if step > diff:
                    idx = start_idx + (step-diff)*stride
            else:
                seq_lengths[b] = ((end_idx-start_idx)//stride)+1
                idx = start_idx + step*stride

            if idx > end_idx:
                attr.append(None)
                x[b,:] = 0.0
                y[b,:] = 0.0
            else:
                assert( idx < self._data_len )
                date = self._dates[idx]
                key = self._keys[idx]
                attr.append((key,date))
                next_idx = idx + forecast_n
                next_key = self._keys[next_idx] if next_idx < len(self._keys) else ""
                x[b,0:len1] = self._get_feature_vector(end_idx,idx)
                if len2 > 0:
                    x[b,len1:len1+len2] = self._get_aux_vector(idx)
                if key == next_key: # targets exist
                    y[b,:] = self._get_feature_vector(end_idx,next_idx)
                else: # no targets exist
                    y[b,:] = None

        return x, y, attr

    def _next_batch(self):
        """
        Generate the next batch of sequences from the data.
        Returns:
          A batch of type Batch (see class def below)
        """
        normalizers = self._get_batch_normalizers()
        seq_lengths = np.full(self._batch_size, self._max_unrollings, dtype=int)
        inputs = list()
        targets = list()
        attribs = list()
        for i in range(self._max_unrollings):
            x, y, attr = self._next_step(i, seq_lengths)
            inputs.append(x)
            targets.append(y)
            attribs.append(attr)

        assert len(inputs) == len(targets)

        ########################################################################
        #   Set cursor for next batch
        ########################################################################
        batch_size = self._batch_size
        num_idxs = len(self._start_indices)
        self._index_cursor = [(self._index_cursor[b]+1)%num_idxs \
                              for b in range(batch_size)]

        return Batch(inputs, targets, attribs, normalizers, seq_lengths)

    def next_batch(self):
        """
        Fetches next batch via the `_next_batch` method, saves batch onto the
        `_batch_cache` attribute list and also returns it.
        """
        b = None

        if self._batch_cache[self._batch_cursor] is not None:
            b = self._batch_cache[self._batch_cursor]
        else:
            b = self._next_batch()
            self._batch_cache[self._batch_cursor] = b

        self._batch_cursor = (self._batch_cursor+1) % (self._num_batches)

        return b

    def get_scaling_params(self, scaler_class):
        if self._scaling_params is None:
            stride = self._stride
            data = self._data
            sample = list()
            z = zip(self._start_indices,self._end_indices)
            indices = random.sample(list(z),
                                    int(0.30*len(self._start_indices)))

            for start_idx, end_idx in indices:
                step = random.randrange(self._min_unrollings)
                cur_idx = start_idx+step*stride
                x1 = self._get_feature_vector(end_idx,cur_idx)
                x2 = self._get_aux_vector(cur_idx)
                sample.append(np.append(x1,x2))

            scaler = None
            if hasattr(sklearn.preprocessing, scaler_class):
                scaler = getattr(sklearn.preprocessing, scaler_class)()
            else:
                raise RuntimeError("Unknown scaler = %s"%scaler_class)

            scaler.fit(sample)

            params = dict()
            params['center'] = scaler.center_ if hasattr(scaler,'center_') else scaler.mean_
            params['scale'] = scaler.scale_

            # Do not scale these features
            for i in self._dont_scale_input_idxs:
                params['center'][i] = 0.0
                params['scale'][i] = 1.0
            
            self._scaling_params = params

        return self._scaling_params

    def get_raw_inputs(self,batch,idx,vec):
        len1 = len(self._fin_colidxs)
        len2 = len(self._aux_colidxs)
        n = batch.normalizers[idx]
        y = vec[0:len1]
        if self._log_squasher is True:
            y = np.multiply(np.sign(y),np.expm1(np.fabs(y)))
        y = n * y
        if len2 > 0 and len(vec) > len1:
            assert(len(vec)==len1+len2)
            y = np.append( y, vec[len1:len1+len2] )
        return y

    def get_raw_outputs(self,batch,idx,vec):
        if self._price_target_idx >= 0:
            return vec
        else:
            return self.get_raw_inputs(batch,idx,vec)

    def _get_cache_filename(self):
        config = self._config
        key_list = list(set(self._data[config.key_field]))
        key_list.sort()
        keys = ''.join(key_list)
        sd = self._start_date if self._start_date is not None else 100001
        ed = self._end_date if self._end_date is not None else 999912
        uid = "%d-%d-%d-%d-%d-%d-%d-%d-%s-%s-%s-%s"%(config.cache_id,sd,ed,
                                                     self._forecast_n,
                                                     self._max_unrollings,
                                                     self._min_unrollings,
                                                     self._stride,self._batch_size,
                                                     config.financial_fields,
                                                     config.aux_fields,
                                                     config.scale_field,
                                                     keys)
        hashed = hashlib.md5(uid.encode()).hexdigest()
        filename = "bcache-%s.pkl"%hashed
        return filename

    def _load_cache(self,verbose=False):
        """
        Caches batches from self by calling the `next_batch()` method (which
        writes batch to the list held by the `_batch_cache` attribute).
        """
        num_batches = self.num_batches
        start_time = time.time()
        if verbose is True:
            print("Caching %d batches ..."%(num_batches),end='')
            sys.stdout.flush()

        self.rewind()
        for i in range(num_batches):
            if verbose is True and (i%(1+num_batches//50))==0:
                print('.',end=''); sys.stdout.flush()
            b = self.next_batch()

        if verbose is True:
            print(" done in %.2f seconds."%(time.time() - start_time))

    def cache(self,verbose=False):
        """
        Caches data if not already cached.
        Does so by either reading cache from a pickled file in the _bcache
        directory, or by loading the cache (via the `_load_cache` method) and
        subsequently writing that to the _bcache directory as a pickled file
        for posterity.
        """
        assert len(self._batch_cache)
        if self._batch_cache[-1] is not None:
            return

        if self._config.cache_id is None: # don't cache
            self._load_cache(verbose)
        else:
            filename = self._get_cache_filename()
            dirname = './_bcache/'
            filename = dirname+filename
            if os.path.isdir(dirname) is not True:
                os.makedirs(dirname)
            if os.path.isfile(filename):
                start_time = time.time()
                if verbose is True:
                    print("Reading cache from %s ..."%filename, end=' ')
                self._batch_cache = pickle.load( open( filename, "rb" ) )
                self._num_batches = len(self._batch_cache)
                if verbose is True:
                    print("done in %.2f seconds."%(time.time() - start_time))
            else:
                self._load_cache(verbose)
                start_time = time.time()
                if verbose is True:
                    print("Writing cache to %s ..."%filename, end=' ')
                pickle.dump(self._batch_cache, open( filename, "wb" ))
                if verbose is True:
                    print("done in %.2f seconds."%(time.time() - start_time))

    def _train_dates(self):
        data = self._data
        dates = list(set(data[self._config.date_field]))
        dates.sort()
        split_date = self._config.split_date
        train_dates = [d for d in dates if d < split_date]
        return train_dates

    def _valid_dates(self):
        data = self._data
        dates = list(set(data[self._config.date_field]))
        dates.sort()
        years = 100*((self._config.min_unrollings*self._config.stride)//12)
        split_date = self._config.split_date - years
        valid_dates = [d for d in dates if d >= split_date]
        return valid_dates

    def train_batches(self, verbose=False):
        """
        Returns a BatchGenerator object built from the subset of self._data that
        corresponds to the 'keys' (uniquely-identified companies) that are _not_
        in the validation set.
        """
        config = self._config
        if config.split_date is not None:
            train_dates = self._train_dates()
            indexes = self._data[config.date_field].isin(train_dates)
            train_data = self._data[indexes]
            if verbose is True:
                print("Training period: %s to %s"%(min(train_dates),max(train_dates)))
        else:
            valid_keys = list(self._validation_set.keys())
            indexes = self._data[config.key_field].isin(valid_keys)
            train_data = self._data[~indexes]
            if verbose is True:
                all_keys = sorted(set(self._data[config.key_field]))
                print("Num training entities: %d"%(len(all_keys)-len(valid_keys)))
        assert(len(train_data))
        return BatchGenerator("", config, validation=False,
                              data=train_data, is_training_only=True)

    def valid_batches(self, verbose=False):
        """
        Returns a BatchGenerator object built from the subset of self._data that
        corresponds to the 'keys' (uniquely-identified companies) that _are_ in
        the validation set.
        """
        config = self._config
        if config.split_date is not None:
            valid_dates = self._valid_dates()
            indexes = self._data[config.date_field].isin(valid_dates)
            if verbose is True:
                print("Validation period: %s to %s"%(min(valid_dates),
                                                     max(valid_dates)))
        else:
            valid_keys = list(self._validation_set.keys())
            indexes = self._data[config.key_field].isin(valid_keys)
            if verbose is True:
                print("Num validation entities: %d"%len(valid_keys))

        valid_data = self._data[indexes]
        assert(len(valid_data))
        return BatchGenerator("", config, validation=False,
                              data=valid_data)


    def shuffle(self):
        # We cannot shuffle until the entire dataset is cached
        if (self._batch_cache[-1] is not None):
            random.shuffle(self._batch_cache)
            self._batch_cusror = 0

    def rewind(self):
        """
        Resets _batch_cursor index to ensure we're working with the first batch.
        """
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
    def max_unrollings(self):
        return self._max_unrollings

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_outputs(self):
        return self._num_outputs

class Batch(object):
    def __init__(self, inputs, targets, attribs, normalizers, seq_lengths):
        self._inputs = inputs
        self._targets = targets
        self._attribs = attribs
        self._normalizers = normalizers
        self._seq_lengths = seq_lengths

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
    def size(self):
        return len(self._seq_lengths)

    @property
    def normalizers(self):
        return self._normalizers

    @property
    def seq_lengths(self):
        return self._seq_lengths

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

import os
import sys

import numpy as np
import tensorflow as tf

class BaseModel(object):
    """
    """

    def train_step(self, sess, batch, keep_prob=1.0):
        """
        Take one step through the data set. A step contains a sequences of batches
        where the sequence is of size max_unrollings. The batches are size
        batch_size.
        Args:
          sess: current tf session the model is being run in
          batch: batch of data of type Batch (see batch_generator.py)
          keep_prob: keep_prob for dropout
        Returns:
           mse: mean squared error scalar for batch
           predictions: the model predictions for each data point in batch
        """

        feed_dict = self._get_feed_dict(batch,keep_prob=keep_prob,training=True)

        (mse, _) = sess.run([self._mse,self._train_op],feed_dict)
        # assert( train_evals > 0 )

        return mse

    def step(self, sess, batch):
        """
        Take one step through the data set. A step contains a sequences of batches
        where the sequence is of size max_unrollings. The batches are size
        batch_size.
        Args:
          sess: current tf session the model is being run in
          batch: batch of data of type Batch
        Returns:
          mse: mean squared error scalar for batch
          predictions: the model predictions for each data point in batch
        """

        feed_dict = self._get_feed_dict(batch,keep_prob=1.0,training=False)

        (mse, preds) = sess.run([self._mse,self._predictions],feed_dict)

        return mse, preds

    def debug_step(self, sess, batch, training=False):
        """
        Take one step through the data set. A step contains a sequences of batches
        where the sequence is of size max_unrollings. The batches are size
        batch_size.
        Args:
          sess: current tf session the model is being run in
          batch: batch of data of type Batch (see batch_generator.py)
          keep_prob: keep_prob for dropout
        Returns:
           mse: mean squared error scalar for batch
           predictions: the model predictions for each data point in batch
        """

        print()
        print("////////////////////////////////////////////////////////////////////////////////////")
        print(batch.targets)

        feed_dict = self._get_feed_dict(batch,keep_prob=1.0,training=training)

        (s,t,lt,lkt,lkti,o,lo,lko,lkoi) = sess.run([self._seq_lengths,self._t,self._lt,self._lkt,self._lkti,self._o,self._lo,self._lko,self._lkoi],feed_dict)

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)

        n = batch.normalizers[0]

        def unnorm(n,x):
            return n * np.multiply(np.sign(x),np.expm1(np.fabs(x)))

        print("Seq Len ////////////////////////////////////////////////////////////////////////")
        print(np.array(s))
        print("Targets ////////////////////////////////////////////////////////////////////////")
        print(np.array(t))
        print(" Last Target Targets ////////////////////////////////////////////////////////////////////////")
        print(np.array(lt))
        print("Last k Targets ////////////////////////////////////////////////////////////////////////////")
        print(np.array(lkt))
        print("Last k Targets i /////////////////////////////////////////////////////////////////////")
        print(np.array(lkti))
        print("Outputs ////////////////////////////////////////////////////////////////////////")
        print(np.array(o))
        print(" Last Output ////////////////////////////////////////////////////////////////////////")
        print(np.array(lo))
        print("Last k Outputs ////////////////////////////////////////////////////////////////////////////")
        print(np.array(lko))
        print("Last k Outputs i /////////////////////////////////////////////////////////////////////")
        print(np.array(lkoi))
        print("////////////////////////////////////////////////////////////////////////////////////")

        (mse, preds) = sess.run([self._mse,self._predictions],feed_dict)
        # assert( train_evals > 0 )

        return mse, preds

    def _get_feed_dict(self, batch, keep_prob=1.0, training=False):

        feed_dict = dict()

        feed_dict[self._batch_size] = batch.inputs[0].shape[0]
        feed_dict[self._seq_lengths] = batch.seq_lengths
        feed_dict[self._keep_prob] = keep_prob
        feed_dict[self._phase] = 1 if training is True else 0

        for i in range(self._max_unrollings):
            feed_dict[self._inputs[i]]  = batch.inputs[i]
            feed_dict[self._targets[i]] = batch.targets[i]

        return feed_dict

    def _center_and_scale(self, x):
            # only center/scale up to the width of x
        n = tf.shape(x)[1]
        return tf.divide( x - self._center[:n], self._scale[:n] )

    def _reverse_center_and_scale(self, x):
        # only center/scale up to the width of x
        n = tf.shape(x)[1]
        return tf.multiply( x, self._scale[:n] ) + self._center[:n]

    def set_scaling_params(self,session,center=None,scale=None):
        assert(center is not None)
        assert(scale is not None)
        session.run(tf.assign(self._center,center))
        session.run(tf.assign(self._scale,scale))

    def set_learning_rate(self, session, lr_value):
        session.run(tf.assign(self._lr, lr_value))
        return lr_value

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def max_unrollings(self):
        return self._max_unrollings

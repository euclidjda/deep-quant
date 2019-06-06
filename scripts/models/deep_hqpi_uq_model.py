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

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import math_ops

from models.base_model import BaseModel


class DeepHqpiUQModel(BaseModel):
    """
    A Deep Rnn Model that supports regression with
    arbitrary number of fixed width hidden layers.

    HQPI : High Quality Prediction Intervals
    Model Type: Prediction Interval Estimation or PIE

    """

    def __init__(self, config):
        """
        Initialize the model
        Args:
          config
        """
        self.config = config
        self._max_unrollings = max_unrollings = config.max_unrollings
        self._min_unrollings = min_unrollings = config.min_unrollings
        self._num_inputs = num_inputs = config.num_inputs
        self._num_outputs = num_outputs = config.num_outputs
        num_hidden = config.num_hidden

        # input/target normalization params
        self._center = tf.get_variable('center', shape=[num_inputs], trainable=False)
        self._scale = tf.get_variable('scale', shape=[num_inputs], trainable=False)

        batch_size = self._batch_size = tf.placeholder(tf.int32, shape=[])
        self._seq_lengths = tf.placeholder(tf.int32, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32, shape=[])
        self._phase = tf.placeholder(tf.bool, name='phase')

        self._inputs = list()
        self._targets = list()

        for _ in range(max_unrollings):
            inp = tf.placeholder(tf.float32, shape=[None, num_inputs])
            tar = tf.placeholder(tf.float32, shape=[None, num_outputs])
            self._inputs.append(inp)
            self._targets.append(tar)

        self._scaled_inputs = [None] * max_unrollings
        self._scaled_targets = [None] * max_unrollings

        for i in range(max_unrollings):
            if config.data_scaler is not None:
                self._scaled_inputs[i] = self._center_and_scale(self._inputs[i])
            else:
                self._scaled_inputs[i] = self._inputs[i]
            if config.data_scaler is not None and config.scale_targets is True:
                self._scaled_targets[i] = self._center_and_scale(self._targets[i])
            else:
                self._scaled_targets[i] = self._targets[i]

        hkp = self._keep_prob if config.hidden_dropout is True else 1.0
        ikp = self._keep_prob if config.input_dropout is True else 1.0
        rkp = self._keep_prob if config.rnn_dropout is True else 1.0

        def rnn_cell():
            cell = None
            if config.rnn_cell == 'gru':
                cell = tf.contrib.rnn.GRUCell(num_hidden)
            elif config.rnn_cell == 'lstm':
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_hidden,
                                                             dropout_keep_prob=rkp,
                                                             dropout_prob_seed=config.seed)
            assert (cell is not None)
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 output_keep_prob=hkp,
                                                 input_keep_prob=ikp, seed=config.seed)
            return cell

        stacked_rnn = tf.contrib.rnn.MultiRNNCell([rnn_cell()
                                                   for _ in range(config.num_layers)])

        rnn_outputs, state = tf.contrib.rnn.static_rnn(stacked_rnn,
                                                       self._scaled_inputs,
                                                       dtype=tf.float32,
                                                       sequence_length=self._seq_lengths)

        # Define Lower Bound output
        self._lb = output_lb_w = tf.get_variable("output_lb_w", [num_hidden, num_outputs])
        output_lb_b = tf.get_variable("output_lb_b", [num_outputs])

        # Define Upper Bound output
        self._ub = output_ub_w = tf.get_variable("output_ub_w", [num_hidden, num_outputs])
        output_ub_b = tf.get_variable("output_ub_b", [num_outputs])

        self._outputs_lb = list()
        self._outputs_ub = list()
        for i in range(max_unrollings):
            output_lb = tf.nn.xw_plus_b(rnn_outputs[i], output_lb_w, output_lb_b)
            output_ub = tf.nn.xw_plus_b(rnn_outputs[i], output_ub_w, output_ub_b)

            # TODO: What is this?
            if config.direct_connections is True:
                self._outputs += self._scaled_inputs[i][:, :num_outputs]

            self._outputs_lb.append(output_lb)
            self._outputs_ub.append(output_ub)

        seqmask = tf.sequence_mask(self._seq_lengths * num_outputs,
                                   max_unrollings * num_outputs, dtype=tf.float32)
        outputs_lb = tf.concat(self._outputs_lb, 1)
        outputs_ub = tf.concat(self._outputs_ub, 1)
        targets = tf.concat(self._scaled_targets, 1)

        seqmask = tf.reshape(seqmask, [batch_size, max_unrollings, num_outputs])
        outputs_lb = tf.reshape(outputs_lb, [batch_size, max_unrollings, num_outputs])
        outputs_ub = tf.reshape(outputs_ub, [batch_size, max_unrollings, num_outputs])
        targets = tf.reshape(targets, [batch_size, max_unrollings, num_outputs])

        outputs_lb = tf.multiply(seqmask, outputs_lb)
        outputs_ub = tf.multiply(seqmask, outputs_ub)
        targets = tf.multiply(seqmask, targets)

        last_k_seqmask = seqmask[:, min_unrollings - 1:, :]
        last_k_outputs_lb = outputs_lb[:, min_unrollings - 1:, :]
        last_k_outputs_ub = outputs_ub[:, min_unrollings - 1:, :]
        last_k_targets = targets[:, min_unrollings - 1:, :]

        reversed_outputs_lb = tf.reverse_sequence(outputs_lb, self._seq_lengths, seq_axis=1, batch_axis=0)

        reversed_outputs_ub = tf.reverse_sequence(outputs_ub, self._seq_lengths, seq_axis=1, batch_axis=0)

        reversed_targets = tf.reverse_sequence(targets,
                                               self._seq_lengths, seq_axis=1, batch_axis=0)
        reversed_seqmask = tf.reverse_sequence(seqmask,
                                               self._seq_lengths, seq_axis=1, batch_axis=0)

        last_output_lb = tf.unstack(reversed_outputs_lb, axis=1)[0]
        last_output_ub = tf.unstack(reversed_outputs_ub, axis=1)[0]
        last_target = tf.unstack(reversed_targets, axis=1)[0]
        last_seqmask = tf.unstack(reversed_seqmask, axis=1)[0]

        if config.data_scaler is not None and config.scale_targets is True:
            self._predictions_lb = self._reverse_center_and_scale(last_output_lb)
            self._predictions_ub = self._reverse_center_and_scale(last_output_ub)
        else:
            self._predictions_lb = last_output_lb
            self._predictions_ub = last_output_ub

        ktidx = config.target_idx

        # For debugging from base_model.debug_step()
        self._lt = last_target
        self._lo_lb = last_output_lb
        self._lo_ub = last_output_ub
        self._lkt = last_k_targets
        self._lko_lb = last_k_outputs_lb
        self._lko_ub = last_k_outputs_ub
        self._lkti = last_k_targets[:, :, ktidx]
        self._lkoi_lb = last_k_outputs_lb[:, :, ktidx]
        self._lkoi_ub = last_k_outputs_ub[:, :, ktidx]
        self._t = targets
        self._o_lb = outputs_lb
        self._o_ub = outputs_ub

        # Different components of MPIW, PICP
        smoothing_pi_check = config.smoothing_pi_check
        confidence_alpha = config.confidence_alpha

        if config.backfill is True:
            self._mpiw_0, self._picp_loss_0, _ = self.get_loss_comps(last_output_lb[:, ktidx], last_output_ub[:, ktidx],
                                                                     last_target[:, ktidx], last_seqmask[:, ktidx],
                                                                     smoothing_pi_check, confidence_alpha)

            self._mpiw_1, self._picp_loss_1, _ = self.get_loss_comps(last_output_lb, last_output_ub, last_target,
                                                                     last_seqmask, smoothing_pi_check, confidence_alpha)
        else:
            self._mpiw_0, self._picp_loss_0, _ = self.get_loss_comps(last_k_outputs_lb[:, :, ktidx],
                                                                     last_k_outputs_ub[:, :, ktidx],
                                                                     last_k_targets[:, :, ktidx],
                                                                     last_k_seqmask[:, :, ktidx],
                                                                     smoothing_pi_check, confidence_alpha)

            self._mpiw_1, self._picp_loss_1, _ = self.get_loss_comps(last_k_outputs_lb, last_k_outputs_ub,
                                                                     last_k_targets, last_k_seqmask,
                                                                     smoothing_pi_check, confidence_alpha)

        self._mpiw_2, self._picp_loss_2, _ = self.get_loss_comps(outputs_lb, outputs_ub, targets, seqmask,
                                                                 smoothing_pi_check, confidence_alpha)

        # MPIW, PICP from last outputs for the given target field
        self._mpiw, self._picp_loss, self._picp = self.get_loss_comps(last_output_lb[:, ktidx],
                                                                      last_output_ub[:, ktidx],
                                                                      last_target[:, ktidx],
                                                                      last_seqmask[:, ktidx],
                                                                      smoothing_pi_check, confidence_alpha)

        # here is the learning part of the graph
        p1 = config.target_lambda
        p2 = config.rnn_lambda
        l2 = config.l2_alpha * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if "_b" not in tf_var.name)

        # Different components of loss
        loss_0 = self._mpiw_0 + config.picp_lambda*self._picp_loss_0
        loss_1 = self._mpiw_1 + config.picp_lambda*self._picp_loss_1
        loss_2 = self._mpiw_2 + config.picp_lambda*self._picp_loss_2

        loss = p1 * loss_0 + (1.0 - p1) * (p2 * loss_1 + (1.0 - p2) * loss_2) + l2
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)

        if config.max_grad_norm > 0:
            grads, self._grad_norm = tf.clip_by_global_norm(grads, config.max_grad_norm)
        else:
            self._grad_norm = tf.constant(0.0)

        self._lr = tf.Variable(0.0, trainable=False)
        optimizer = None
        args = config.optimizer_params
        if hasattr(tf.train, config.optimizer):
            optimizer = getattr(tf.train, config.optimizer)(learning_rate=self._lr, **args)
        else:
            raise RuntimeError("Unknown optimizer = %s" % config.optimizer)

        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def get_loss_comps(self, outputs_lb, outputs_ub, targets, mask, s, alpha):
        """ Return tensor k_soft. k_soft calculates which prediction intervals contain true values and applies
        smoothing s

        :param outputs_lb: lower bound outputs
        :param outputs_ub: upper bound outputs
        :param targets: targets
        :param mask: sequence mask tensor
        :param s: smoothing parameter
        :param alpha: alpha to calculate confidence level. alpha = 0.05 is 95% confidence level
        :return mpiw: mean prediction interval width
        :return picp_loss: prediction interval coverage probability loss term
        :return picp: prediction interval coverage probability
        """

        # Soft check
        lb_check_soft = tf.nn.sigmoid(s*(targets - outputs_lb))
        ub_check_soft = tf.nn.sigmoid(s*(outputs_ub - targets))
        pi_check_soft = tf.multiply(lb_check_soft, ub_check_soft)

        n = tf.reduce_sum(mask)
        c = tf.reduce_sum(pi_check_soft)

        mpiw = (1/c)*tf.reduce_sum(tf.multiply(tf.subtract(outputs_ub, outputs_lb), pi_check_soft))
        picp = tf.divide(c, n)
        picp_loss = tf.multiply(n/(alpha*(1-alpha)), tf.maximum(0.0, 1-alpha - picp)**2)
        return mpiw, picp_loss, picp


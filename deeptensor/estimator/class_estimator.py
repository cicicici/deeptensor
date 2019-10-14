from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch

from deeptensor.estimator import estimator as estimator


class ClassEstimator(estimator.BaseEstimator):

    def __init__(self, opt, cfg):
        super(ClassEstimator, self).__init__(opt, cfg)
        dt.debug(dt.DC.TRAIN, "[EST] {} initialized".format(type(self).__name__))

    def preprocess_data(self, tensor, is_training):
        return None

    def build_data(self, is_training):
        args = self._opt.args

        # Params
        # args.batch_size
        # args.valid_size
        # distorted
        # out_height, out_width
        # data_format

        # Output
        data = None
        ep_size = 0 # data.train.num_batch
        v_ep_size = 0 # data.valid.num_batch

        return dt.Opt(data=data, ep_size=ep_size, v_ep_size=v_ep_size)

    def build_model(self):
        # Params
        # args.model_name == "resnet":
        # args.model_type == "v1":
        # args.class_num
        # args.block_type
        # args.blocks
        # args.shortcut
        # args.regularizer
        # args.conv_decay
        # args.fc_decay
        # self._opt.data_format
        # args.se_ratio)

        return None

    def forward(self, tensor, is_training, reuse=False):
        args = self._opt.args
        logits = None
        return logits

    def define_loss(self, logits, labels, is_training):
        # Params
        # softmax

        loss = None # dt.loss.ce(logits, target=labels, softmax=False)
        return loss

    def define_validation(self):
        valid_metric = []
        args = self._opt.args
        #if args.validate_ep > 0:
        #    vx = self._opt.data.vx
        #    vy = self._opt.data.vy
            # May not need this ctx line
        #    with dt.ctx(is_training=False, reuse=True, summary=False):
        #        logits_val = self.forward(vx, False, reuse=True)
        #    loss_val = dt.loss.ce(logits_val, target=vy, softmax=False)
        #    soft_val = dt.activation.softmax(logits_val)
        #    acc1_val = tf.reduce_mean(dt.metric.accuracy(soft_val, target=vy), name='acc1')
        #    valid_metric.append(dt.Opt(name='valid', ops=[loss_val, acc1_val], cnt=self._opt.data.v_ep_size))
        return valid_metric

    def pre_train(self):
        print(self._opt)
        return None

    def post_train(self):
        return None


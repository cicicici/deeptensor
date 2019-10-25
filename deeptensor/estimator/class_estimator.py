from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deeptensor.estimator import estimator as estimator


class ClassEstimator(estimator.BaseEstimator):

    def __init__(self, opt, cfg):
        super(ClassEstimator, self).__init__(opt, cfg)
        self.tag = "EST::CLASS"
        dt.trace(dt.DC.MODEL, "[{}] ({}) __init__".format(self.tag, type(self).__name__))

    def preprocess_data(self, tensor, is_training):
        return None

    def build_data(self):
        args = self._ctx.args

        # Params
        # args.batch_size
        # args.valid_size
        # distorted
        # out_height, out_width
        # data_format

        self._data = None
        return False

    def load_data(self):
        self.data.load_data()
        return True

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

        self._model = None
        return False

    def build_optimizer(self):
        self._optimizer = optim.SGD(self._model.parameters(), lr=dt.train.get_lr_val(), momentum=self._ctx.momentum)
        return True

    def build_hooks(self):
        return True

    def forward(self, tensor, is_training):
        logits = self._model(tensor)
        return logits

    def loss(self, logits, labels, is_training):
        loss = F.nll_loss(logits, labels)
        return loss

    def pred(self, logits, is_training):
        pred = logits.argmax(dim=1, keepdim=True)
        return pred

    def correct(self, logits, labels, is_training):
        pred = self.pred(logits, is_training)
        correct = pred.eq(labels.view_as(pred))
        return correct

    def acc(self, logits, labels, is_training):
        correct = self.correct(logits, labels, is_training)
        acc = correct.sum().item() / len(labels)
        return acc

    def pre_train(self):
        dt.info(dt.DC.TRAIN, 'pre train [{}] device: {}'.format(self.tag, self._device))
        return None

    def post_train(self):
        return None


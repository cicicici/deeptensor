from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from deeptensor.estimator import BaseEstimator


class ClassEstimator(BaseEstimator):

    def __init__(self, ctx):
        super(ClassEstimator, self).__init__(ctx)
        self.tag = "EST::CLASS"
        dt.trace(dt.DC.MODEL, "[{}] ({}) __init__".format(self.tag, type(self).__name__))

    def preprocess_data(self, tensor, is_training):
        return None

    def build_data(self):
        #args = self._ctx.args

        # Params
        # args.batch_size
        # args.valid_size
        # distorted
        # out_height, out_width
        # data_format

        self._data = None
        return False

    def build_model(self):
        # Params
        # args.model_name == "resnet":
        # args.model_type == "v1":
        # args.class_num
        # args.block_type
        # args.blocks
        # args.regularizer
        # self._opt.data_format

        self._model = None
        return False

    def build_criterion(self):
        self._criterion = nn.CrossEntropyLoss()

        #def nll_loss_fn(logits, labels):
        #    return F.nll_loss(F.log_softmax(logits, dim=1), labels)
        #self._criterion = nll_loss_fn

        return True

    def build_optimizer(self):
        self._optimizer = optim.SGD(self._model.parameters(), lr=self.trainer.get_lr_val(), momentum=self._ctx.momentum, weight_decay=self._ctx.weight_decay)
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

    def metric(self, logits, labels, is_training):
        if is_training:
            correct = self.correct(logits, labels, is_training)
            acc = correct.float().sum().div_(len(labels))
            return [dt.Opt(name='top1', tensor=acc)]
        else:
            acc = dt.metric.accuracy(logits, labels, topk=(1, 5))
            return [dt.Opt(name='top1', tensor=acc[0]),
                    dt.Opt(name='top5', tensor=acc[1])]

    def pre_train(self):
        dt.info(dt.DC.TRAIN, 'pre train [{}] device: {}'.format(self.tag, self.trainer.device))
        return None

    def post_train(self):
        return None

    def post_model(self):
        if dt.train.is_chief():
            #dt.info(dt.DC.TRAIN, "\n{}".format(self._model))
            dt.summary.summary_model_patch(self._model)
            dt.info(dt.DC.TRAIN, "\n{}".format(dt.summary.summary_model_fwd(self._model, (3, 32, 32), device='cpu')))
            dt.summary.summary_model_patch(self._model, patch_fn=dt.summary.patch_clear_dt)

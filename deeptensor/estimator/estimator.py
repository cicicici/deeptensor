from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod

import deeptensor as dt
import torch
import horovod.torch as hvd


class BaseEstimator(object):
    __metaclass__ = ABCMeta

    def __init__(self, opt, cfg):
        self.tag = "EST::BASE"
        dt.trace(dt.DC.MODEL, "[{}] ({}) __init__".format(self.tag, type(self).__name__))
        self._ctx = opt
        self._cfg = cfg

        self._data = None
        self._model = None
        self._criterion = None
        self._optimizer = None
        self._train_hooks = []
        self._valid_hooks = []

        self._use_cuda = dt.train.use_cuda()
        self._device = dt.train.device()

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = value

    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def device(self):
        return self._device

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._model

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def train_hooks(self):
        return self._train_hooks

    @property
    def valid_hooks(self):
        return self._valid_hooks

    # Abstract
    @abstractmethod
    def preprocess_data(self, tensor, is_training):
        return None

    @abstractmethod
    def build_data(self):
        return None

    @abstractmethod
    def build_model(self):
        return None

    @abstractmethod
    def build_criterion(self):
        return None

    @abstractmethod
    def build_optimizer(self):
        return None

    @abstractmethod
    def build_hooks(self):
        return None

    @abstractmethod
    def forward(self, tensor, is_training):
        return None

    @abstractmethod
    def pred(self, logits, is_training):
        return None

    @abstractmethod
    def correct(self, logits, labels, is_training):
        return None

    @abstractmethod
    def metric(self, logits, labels, is_training):
        # Must return tensor in format of:
        # return [dt.Opt(name='top1', tensor=acc[0]), dt.Opt(name='top5', tensor=acc[1])]
        return None

    @abstractmethod
    def validation(self):
        return None

    def pre_train(self):
        return False

    def post_train(self):
        return False

    # Core
    def build_estimator(self):
        self.build_data()
        self.build_model()
        self.build_criterion()
        self.build_optimizer()
        self.build_hooks()
        return self

    def inference(self, tensor, checkpoint, batch_size=None):
        return None


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
        self._opt = opt
        self._cfg = cfg

        self._data = None
        self._model = None
        self._loss = None

        self._use_cuda = not cfg.no_cuda and torch.cuda.is_available()
        self._device = torch.device("cuda" if self._use_cuda else "cpu")

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
    def forward(self, tensor, is_training, reuse=False):
        return None

    @abstractmethod
    def define_loss(self, logits, labels, is_training):
        return None

    @abstractmethod
    def define_validation(self):
        return None

    # Hooks
    def build_train_hooks(self, is_chief):
        return []

    def pre_train(self):
        return None

    def post_train(self):
        return None

    # Core
    def build_estimator(self):
        return None

    def train(self):
        return None

    def evaluate(self):
        return None

    def inference(self, tensor, checkpoint, batch_size=None):
        return None


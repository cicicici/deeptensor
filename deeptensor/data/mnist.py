from __future__ import absolute_import, division, print_function, unicode_literals

import deeptensor as dt
import torch


class Mnist(object):

    _data_dir = './_asset/data/mnist'

    ORIG_IMAGE_SIZE = 28
    ORIG_LABEL_SIZE = 1
    NUM_CHANNELS = 1
    IMAGE_SIZE = 28
    NUM_CLASSES = 10
    TRAIN_NUM_PER_EPOCH = 50000
    EVAL_NUM_PER_EPOCH = 10000

    DATA_FORMAT = dt.dformat.NHWC

    def __init__(self, batch_size=128, valid_size=128, reshape=False, one_hot=False, data_format=dt.dformat.DEFAULT):
        self._data_dir = Mnist._data_dir
        self._batch_size = batch_size
        self._valid_size = valid_size
        self._reshape = reshape
        self._one_hot = one_hot
        self._data_format = data_format

    def init_data(self):
        self.train, self.valid, self.test = dt.Opt(), dt.Opt, dt.Opt()

        self.train.num_batch = Mnist.TRAIN_NUM_PER_EPOCH // self._batch_size
        self.valid.num_batch = Mnist.EVAL_NUM_PER_EPOCH // self._valid_size

        return self

    def generate(self):

        #self.train.images, self.train.labels
        #self.valid.images, self.valid.labels

        return self


from __future__ import absolute_import, division, print_function, unicode_literals

import deeptensor as dt
import torch


class Mnist(object):

    ORIG_IMAGE_SIZE = 28
    ORIG_LABEL_SIZE = 1

    NUM_CHANNELS = 1
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    NUM_CLASSES = 10

    TRAIN_NUM_PER_EPOCH = 50000
    EVAL_NUM_PER_EPOCH = 10000

    DATA_FORMAT = dt.dformat.NHWC

    def __init__(self, data_dir = '_asset/data/mnist',
                 batch_size=128, valid_size=128,
                 out_height=IMAGE_HEIGHT, out_width=IMAGE_WIDTH, distorted=False,
                 num_workers=1, pin_memory=True,
                 shard=False, data_format=dt.dformat.DEFAULT):
        self._data_dir = data_dir

        self._batch_size = batch_size
        self._valid_size = valid_size

        self._out_height = out_height
        self._out_width = out_width
        self._distorted = distorted

        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._shard = shard
        self._data_format = data_format

    def init_data(self):
        self.train, self.valid, self.test = dt.Opt(), dt.Opt, dt.Opt()

        self.train.num_batch = Mnist.TRAIN_NUM_PER_EPOCH // self._batch_size
        self.valid.num_batch = Mnist.EVAL_NUM_PER_EPOCH // self._valid_size

        return self

    def generate(self):

        return self


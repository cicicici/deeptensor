from __future__ import absolute_import, division, print_function, unicode_literals

import math

import deeptensor as dt
import torch
from torchvision import datasets, transforms

from deeptensor.data import BaseData

class Mnist(BaseData):

    ORIG_IMAGE_SIZE = 28
    ORIG_LABEL_SIZE = 1

    NUM_CHANNELS = 1
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    NUM_CLASSES = 10

    TRAIN_NUM_PER_EPOCH = 60000
    VALID_NUM_PER_EPOCH = 10000
    TEST_NUM_PER_EPOCH = 10000

    DATA_FORMAT = dt.dformat.NCHW

    def __init__(self, data_dir = '_asset/data/mnist',
                 batch_size=128, valid_size=128,
                 out_height=IMAGE_HEIGHT, out_width=IMAGE_WIDTH, distorted=False,
                 num_workers=1, pin_memory=True,
                 shuffle=True, data_format=dt.dformat.DEFAULT):
        super(Mnist, self).__init__()
        self.tag = "DATA::MNIST"

        self._data_dir = data_dir

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._test_size = valid_size

        self._out_height = out_height
        self._out_width = out_width
        self._distorted = distorted

        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._shuffle = shuffle
        self._data_format = data_format

    def init_data(self):
        dt.trace(dt.DC.DATA, "[{}] init data".format(self.tag))

        self.train, self.valid, self.test = dt.Opt(), dt.Opt, dt.Opt()

        self.train.batch_size = self._batch_size
        self.valid.batch_size = self._valid_size
        self.test.batch_size = self._test_size

        self.train.num_total = Mnist.TRAIN_NUM_PER_EPOCH
        self.valid.num_total = Mnist.VALID_NUM_PER_EPOCH
        self.test.num_total = Mnist.TEST_NUM_PER_EPOCH

        self.train.num_batch = int(math.ceil(Mnist.TRAIN_NUM_PER_EPOCH / self._batch_size))
        self.valid.num_batch = int(math.ceil(Mnist.VALID_NUM_PER_EPOCH / self._valid_size))
        self.test.num_batch = int(math.ceil(Mnist.TEST_NUM_PER_EPOCH / self._test_size))

        return self

    def load_data(self):
        dt.trace(dt.DC.DATA, "[{}] load data".format(self.tag))

        kwargs = {'num_workers': 1, 'pin_memory': True} if self._pin_memory else {}
        self.train.dataset = datasets.MNIST(self._data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        self.train.loader = torch.utils.data.DataLoader(self.train.dataset,
            batch_size=self._batch_size, shuffle=self._shuffle, **kwargs)

        self.valid.dataset = datasets.MNIST(self._data_dir, train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        self.valid.loader = torch.utils.data.DataLoader(self.valid.dataset,
            batch_size=self._valid_size, shuffle=False, **kwargs)

        self.test.dataset = datasets.MNIST(self._data_dir, train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        self.test.loader = torch.utils.data.DataLoader(self.test.dataset,
            batch_size=self._test_size, shuffle=False, **kwargs)

        return self

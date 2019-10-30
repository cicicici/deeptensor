from __future__ import absolute_import, division, print_function, unicode_literals

import math

import deeptensor as dt
import torch
from torchvision import datasets, transforms

from deeptensor.data import data as data

class Cifar10(data.BaseData):

    ORIG_IMAGE_SIZE = 32
    ORIG_LABEL_SIZE = 1

    NUM_CHANNELS = 3
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    NUM_CLASSES = 10

    TRAIN_NUM_PER_EPOCH = 50000
    VALID_NUM_PER_EPOCH = 10000
    TEST_NUM_PER_EPOCH = 10000

    DATA_FORMAT = dt.dformat.NCHW

    def __init__(self, data_dir = '_asset/data/cifar10',
                 batch_size=128, valid_size=128,
                 out_height=IMAGE_HEIGHT, out_width=IMAGE_WIDTH, distorted=False,
                 num_workers=1, pin_memory=True,
                 shuffle=True, data_format=dt.dformat.DEFAULT):
        super(Cifar10, self).__init__()
        self.tag = "DATA::CIFAR10"

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

        self.train.num_total = Cifar10.TRAIN_NUM_PER_EPOCH
        self.valid.num_total = Cifar10.VALID_NUM_PER_EPOCH
        self.test.num_total = Cifar10.TEST_NUM_PER_EPOCH

        self.train.num_batch = int(math.ceil(Cifar10.TRAIN_NUM_PER_EPOCH / self._batch_size))
        self.valid.num_batch = int(math.ceil(Cifar10.VALID_NUM_PER_EPOCH / self._valid_size))
        self.test.num_batch = int(math.ceil(Cifar10.TEST_NUM_PER_EPOCH / self._test_size))

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return self

    def load_data(self):
        dt.trace(dt.DC.DATA, "[{}] load data".format(self.tag))

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        kwargs = {'num_workers': self._num_workers, 'pin_memory': True} if self._pin_memory else {}
        self.train.dataset = datasets.CIFAR10(self._data_dir, train=True, download=True, transform=transform_train)
        if dt.train.is_mp():
            # Horovod: use DistributedSampler to partition the training data.
            self.train.sampler = torch.utils.data.distributed.DistributedSampler(
                self.train.dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=self._shuffle)
            self.train.loader = torch.utils.data.DataLoader(self.train.dataset,
                batch_size=self._batch_size, shuffle=False, sampler=self.train.sampler, **kwargs)
        else:
            self.train.loader = torch.utils.data.DataLoader(self.train.dataset,
                batch_size=self._batch_size, shuffle=self._shuffle, **kwargs)

        self.valid.dataset = datasets.CIFAR10(self._data_dir, train=False, transform=transform_test)
        if dt.train.is_mp():
            # Horovod: use DistributedSampler to partition the validation data.
            self.valid.sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid.dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
            self.valid.loader = torch.utils.data.DataLoader(self.valid.dataset,
                batch_size=self._batch_size, shuffle=False, sampler=self.valid.sampler, **kwargs)
        else:
            self.valid.loader = torch.utils.data.DataLoader(self.valid.dataset,
                batch_size=self._valid_size, shuffle=False, **kwargs)

        self.test.dataset = datasets.CIFAR10(self._data_dir, train=False, transform=transform_test)
        if dt.train.is_mp():
            # Horovod: use DistributedSampler to partition the test data.
            self.test.sampler = torch.utils.data.distributed.DistributedSampler(
                self.test.dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
            self.test.loader = torch.utils.data.DataLoader(self.test.dataset,
                batch_size=self._batch_size, shuffle=False, sampler=self.test.sampler, **kwargs)
        else:
            self.test.loader = torch.utils.data.DataLoader(self.test.dataset,
                batch_size=self._test_size, shuffle=False, **kwargs)

        return self


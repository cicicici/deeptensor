from __future__ import absolute_import, division, print_function, unicode_literals

import os
import math
import PIL

import deeptensor as dt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import horovod.torch as hvd

from deeptensor.data import BaseData


class ImageNet(BaseData):

    NUM_CHANNELS = 3
    IMAGE_SIZE = 224
    CROP_RATIO = 0.875
    NUM_CLASSES = 1000

    TRAIN_NUM_PER_EPOCH = 1281167
    VALID_NUM_PER_EPOCH = 50000
    TEST_NUM_PER_EPOCH = 50000

    DATA_FORMAT = dt.dformat.NCHW

    TRAIN_DIR = 'train'
    VALIDATION_DIR = 'valid'
    TEST_DIR = 'valid'

    MEAN_RGB = [0.485, 0.456, 0.406]
    VAR_RGB = [0.229, 0.224, 0.225]

    def __init__(self, data_dir = '/datasets/imagenet',
                 batch_size=32, valid_size=32,
                 out_size=IMAGE_SIZE, crop_ratio=CROP_RATIO,
                 num_workers=1, pin_memory=True,
                 shuffle=True, data_format=dt.dformat.DEFAULT):
        super(ImageNet, self).__init__()
        self.tag = "DATA::IMAGENET"

        self._data_dir = data_dir

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._test_size = valid_size

        self._out_size = out_size
        self._crop_ratio = crop_ratio

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

        self.train.num_total = ImageNet.TRAIN_NUM_PER_EPOCH
        self.valid.num_total = ImageNet.VALID_NUM_PER_EPOCH
        self.test.num_total = ImageNet.TEST_NUM_PER_EPOCH

        self.train.num_batch = int(math.ceil(ImageNet.TRAIN_NUM_PER_EPOCH / self._batch_size / hvd.size()))
        self.valid.num_batch = int(math.ceil(ImageNet.VALID_NUM_PER_EPOCH / self._valid_size / hvd.size()))
        self.test.num_batch = int(math.ceil(ImageNet.TEST_NUM_PER_EPOCH / self._test_size / hvd.size()))

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return self

    def load_data(self):
        dt.trace(dt.DC.DATA, "[{}] load data".format(self.tag))

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self._out_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=ImageNet.MEAN_RGB, std=ImageNet.VAR_RGB),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(int(self._out_size / self._crop_ratio), interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(self._out_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=ImageNet.MEAN_RGB, std=ImageNet.VAR_RGB),
        ])

        kwargs = {'num_workers': self._num_workers, 'pin_memory': True} if self._pin_memory else {}

        train_dataset_root = os.path.join(self._data_dir, ImageNet.TRAIN_DIR)
        self.train.dataset = datasets.ImageFolder(root=train_dataset_root, transform=transform_train)
        if dt.train.is_mp():
            # Horovod: use DistributedSampler to partition the training data.
            self.train.sampler = torch.utils.data.distributed.DistributedSampler(
                self.train.dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=self._shuffle)
            self.train.loader = torch.utils.data.DataLoader(self.train.dataset,
                batch_size=self._batch_size, shuffle=False, sampler=self.train.sampler, **kwargs)
        else:
            self.train.loader = torch.utils.data.DataLoader(self.train.dataset,
                batch_size=self._batch_size, shuffle=self._shuffle, **kwargs)

        valid_dataset_root = os.path.join(self._data_dir, ImageNet.VALIDATION_DIR)
        self.valid.dataset = datasets.ImageFolder(root=valid_dataset_root, transform=transform_test)
        if dt.train.is_mp():
            # Horovod: use DistributedSampler to partition the validation data.
            self.valid.sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid.dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
            self.valid.loader = torch.utils.data.DataLoader(self.valid.dataset,
                batch_size=self._valid_size, shuffle=False, sampler=self.valid.sampler, **kwargs)
        else:
            self.valid.loader = torch.utils.data.DataLoader(self.valid.dataset,
                batch_size=self._valid_size, shuffle=False, **kwargs)

        test_dataset_root = os.path.join(self._data_dir, ImageNet.TEST_DIR)
        self.test.dataset = datasets.ImageFolder(root=test_dataset_root, transform=transform_test)
        if dt.train.is_mp():
            # Horovod: use DistributedSampler to partition the test data.
            self.test.sampler = torch.utils.data.distributed.DistributedSampler(
                self.test.dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
            self.test.loader = torch.utils.data.DataLoader(self.test.dataset,
                batch_size=self._test_size, shuffle=False, sampler=self.test.sampler, **kwargs)
        else:
            self.test.loader = torch.utils.data.DataLoader(self.test.dataset,
                batch_size=self._test_size, shuffle=False, **kwargs)

        return self

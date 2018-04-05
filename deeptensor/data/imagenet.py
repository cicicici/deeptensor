from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import deeptensor as dt
import tensorflow as tf


_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_NUM_VALID_FILES = 128

def get_filenames(is_training, data_dir):
    if is_training:
        return [
            os.path.join(data_dir, 'train-%05d-of-01024' % (i+1))
            for i in range(_NUM_TRAIN_FILES)]
    else:
        return [
            os.path.join(data_dir, 'validation-%05d-of-00128' % (i+1))
            for i in range(_NUM_VALID_FILES)]

def _distort_image(image, label, is_training):
    image = dt.data.vgg_processing.preprocess_image(image, 224, 224, is_training=is_training)
    #if is_training:
    #    image, label = dt.data.image_processing.distort_image(image, label, height=224, width=224, resize=False, flip=False, color=True)
    return image, label

class ImageNet(object):

    _data_dir = './_asset/imagenet'

    def __init__(self, data_dir, data_type='tfrecord', batch_size=32, valid_size=100, preproc_threads=4, splits=1,
                 shuffle=True, shuffle_size=0, shard=True, distorted=True, class_num=1000, class_min=0):
        self._data_dir = data_dir
        self._data_type = data_type
        self._batch_size = batch_size
        self._valid_size = valid_size
        self._preproc_threads = preproc_threads * splits
        self._splits = splits
        self._shuffle = shuffle
        self._shuffle_size = shuffle_size
        self._shard = shard
        self._distorted = distorted

        self._class_num = class_num
        self._class_min = class_min

    def tfrecord_data(self, is_training, shuffle=False):
        filenames = get_filenames(is_training, '{}/train-val-tfrecord-480'.format(self._data_dir))
        if is_training:
            num_images = _NUM_IMAGES['train']
            b_size = self._batch_size
        else:
            num_images = _NUM_IMAGES['validation']
            b_size = self._valid_size

        tfrecord = dt.data.ImageTFRecord(filenames, num_images, self._class_num, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS,
                                         batch_size=b_size, shuffle=shuffle, shuffle_size=self._shuffle_size, epochs=10000, shard=(self._shard and is_training),
                                         is_training=is_training, distorted=self._distorted, one_hot=False, preproc_threads=self._preproc_threads).init_data()
        dt.debug(dt.DC.DATA, 'TFRecord: training {}, images {}, batches {}, batch_size {}'
                                 .format(is_training, tfrecord._num_images, tfrecord._num_batch, b_size))

        num_images = tfrecord._num_images

        return tfrecord, num_images

    def folder_data(self, is_training, shuffle=False):

        if is_training:
            idx_file = 'train.txt'
            subdir = 'train'
            b_size = self._batch_size
        else:
            idx_file = 'val.txt'
            subdir = 'val'
            b_size = self._valid_size

        folder = dt.data.ImageFolder(self._data_dir,
                                     idx_file,
                                     subdir=subdir,
                                     batch_size=b_size,
                                     preproc_threads=self._preproc_threads,
                                     splits=self._splits,
                                     shuffle=shuffle,
                                     shuffle_size=self._shuffle_size,
                                     distorted=self._distorted,
                                     class_num=self._class_num,
                                     class_min=self._class_min,
                                     distort_image_fn=lambda i, l: _distort_image(i, l, is_training)).init_data()
        dt.debug(dt.DC.DATA, 'Folder: training {}, images {}, batches {}, batch_size {}'
                                 .format(is_training, folder._num_examples_per_epoch, folder._num_batches_per_epoch, b_size))

        num_images = folder._num_examples_per_epoch

        return folder, num_images

    def init_data(self):
        dt.debug(dt.DC.DATA, 'Data dir: {}'
                                 .format(self._data_dir))

        self.train, self.valid = dt.Opt(), dt.Opt()

        if self._data_type == 'tfrecord':
            self.train.data, self.train.num_images = self.tfrecord_data(True, shuffle=self._shuffle)
            self.valid.data, self.valid.num_images = self.tfrecord_data(False, shuffle=False)
        else:
            self.train.data, self.train.num_images = self.folder_data(True, shuffle=self._shuffle)
            self.valid.data, self.valid.num_images = self.folder_data(False, shuffle=False)

        self.train.num_batch = self.train.num_images // self._batch_size
        self.valid.num_batch = self.valid.num_images // self._valid_size

        return self

    def generate(self):
        dt.debug(dt.DC.DATA, 'Data dir: {}'
                                 .format(self._data_dir))

        self.train.data.generate()
        self.valid.data.generate()

        if self._data_type == 'tfrecord':
            self.train.images, self.train.labels = self.train.data._images_batch, self.train.data._labels_batch
            self.valid.images, self.valid.labels = self.valid.data._images_batch, self.valid.data._labels_batch
        else:
            self.train.images, self.train.labels = self.train.data._images_splits[0], self.train.data._labels_splits[0]
            self.valid.images, self.valid.labels = self.valid.data._images_splits[0], self.valid.data._labels_splits[0]

        return self


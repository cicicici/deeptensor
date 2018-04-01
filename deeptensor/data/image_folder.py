from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import random

import deeptensor as dt
import tensorflow as tf


class ImageFolder(object):

    _data_dir = './asset/images'

    def __init__(self, data_dir, idx_file, subdir=None, batch_size=128, preproc_threads=4, splits=1,
                 shuffle=True, shuffle_size=0, distorted=True, class_num=2, class_min=0,
                 distort_image_fn=None, random_seed=12345):
        self._data_dir = data_dir
        self._idx_file = idx_file
        self._subdir = subdir
        self._batch_size = batch_size
        self._preproc_threads = preproc_threads * splits
        self._splits = splits
        self._shuffle = shuffle
        self._shuffle_size = shuffle_size
        self._distorted = distorted

        self._class_num = class_num
        self._class_min = class_min

        if distort_image_fn is None:
            self._distort_image_fn = dt.data.image_processing.distort_image
        else:
            self._distort_image_fn = distort_image_fn

        self._random_seed = random_seed

        self._num_examples_per_epoch = 0
        self._num_batches_per_epoch = 0

        self._image_type = None
        self._files = None
        self._labels = None
        self._dataset = None

        self._images_batch = None
        self._labels_batch = None

        self._images_splits = None
        self._labels_splits = None

    def get_image_type(self, fname):
        name, ext = os.path.splitext(fname)
        if ext.lower() == '.png':
            image_type = 'png'
        elif ext.lower() == '.jpg' or ext.lower() == '.jpeg':
            image_type = 'jpeg'
        return image_type

    def build_file_label_list(self, data_directory):
        filenames = []
        labels = []
        idx_file = os.path.join(data_directory, self._idx_file)
        _, idx_ext = os.path.splitext(idx_file)
        if idx_ext == ".json":
            with open(idx_file, 'r') as data_file:
                data = json.load(data_file)
            for k in data['labelValue']:
                filenames.append(str(os.path.join(data_directory, k + '.png')))
                labels.append(data['labelValue'][k])
            self._image_type = 'png'
        elif idx_ext == ".txt":
            with open(idx_file, 'r') as data_file:
                lines = data_file.readlines()
                for line in lines:
                    split = line.split()
                    if self._subdir is not None:
                        fname = str(os.path.join(data_directory, self._subdir, split[0]))
                    else:
                        fname = str(os.path.join(data_directory, split[0]))
                    label = int(split[1])
                    #dt.log(dt.DC.DATA, dt.DL.DEBUG, "fname: {}, label: {}".format(fname, label))

                    if label >= self._class_min and label < (self._class_min + self._class_num) and os.path.isfile(fname):
                        if self._image_type is None:
                            self._image_type = self.get_image_type(fname)
                        filenames.append(fname)
                        labels.append(label - self._class_min)

        # Shuffle all inputs, critical
        shuffled_index = list(range(len(filenames)))
        random.seed(self._random_seed)
        random.shuffle(shuffled_index)

        rand_filenames = [filenames[i] for i in shuffled_index]
        rand_labels = [labels[i] for i in shuffled_index]

        return rand_filenames, rand_labels

    def create_file_tensor(self):
        self._files, self._labels = self.build_file_label_list(self._data_dir)
        self._num_examples_per_epoch = len(self._files)
        self._num_batches_per_epoch = int(np.ceil(1.0 * self._num_examples_per_epoch / self._batch_size))
        return self

    def create_data_slices(self):
        self._dataset = tf.data.Dataset.from_tensor_slices((self._files, self._labels))
        return self

    def decode_image(self):
        dt.log(dt.DC.DATA, dt.DL.DEBUG, "image type: {}".format(self._image_type))
        if self._image_type == "png":
            self._dataset = self._dataset.map(dt.data.image_processing.parse_filename_example_png,
                                              num_parallel_calls=self._preproc_threads)
        elif self._image_type == "jpeg":
            self._dataset = self._dataset.map(dt.data.image_processing.parse_filename_example_jpeg,
                                              num_parallel_calls=self._preproc_threads)

        return self

    def distort_image(self):
        self._dataset = self._dataset.map(self._distort_image_fn,
                                          num_parallel_calls=self._preproc_threads)
        return self

    def batch_repeat(self):
        if self._shuffle:
            shuffle_size = self._shuffle_size if self._shuffle_size > 0 else self._batch_size * 100
            self._dataset = self._dataset.shuffle(shuffle_size)
        self._dataset = self._dataset.repeat()
        self._dataset = self._dataset.batch(self._batch_size)
        self._dataset = self._dataset.prefetch(buffer_size=self._batch_size * 2)

        train_dataset_iterator = self._dataset.make_one_shot_iterator()
        self._images_batch, self._labels_batch = train_dataset_iterator.get_next()

        return self

    def split(self):
        if self._splits > 1:
            self._images_splits = tf.split(axis=0, num_or_size_splits=self._splits, value=self._images_batch)
            self._labels_splits = tf.split(axis=0, num_or_size_splits=self._splits, value=self._labels_batch)
        else:
            self._images_splits = [self._images_batch]
            self._labels_splits = [self._labels_batch]

        return self

    def init_data(self):
        self.create_file_tensor()
        return self

    def generate(self):
        self.create_data_slices()
        self.decode_image()
        if self._distorted:
            self.distort_image()
        self.batch_repeat()
        self.split()

        return self


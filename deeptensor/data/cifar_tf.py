from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import deeptensor as dt
import tensorflow as tf
import horovod.tensorflow as hvd


class Cifar10TF(object):

    ORIG_IMAGE_HEIGHT = 32
    ORIG_IMAGE_WIDTH = 32
    ORIG_LABEL_SIZE = 1
    NUM_CHANNELS = 3
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    # padding to each side for random cropping
    IMAGE_RAND_CROP_PADDING = 4
    NUM_CLASSES = 10
    TRAIN_NUM_PER_EPOCH = 50000
    EVAL_NUM_PER_EPOCH = 10000

    def __init__(self, data_dir='_asset/data/cifar10',
                 batch_size=128, valid_size=128, distorted=False,
                 out_height=IMAGE_HEIGHT, out_width=IMAGE_WIDTH,
                 preproc_threads=4, shard=True):
        self._data_dir = data_dir

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._distorted = distorted
        self._out_height = out_height
        self._out_width = out_width
        self._preproc_threads = preproc_threads
        self._shard = shard

    def get_filenames(self, subset):
        if subset in ['train', 'validation']:
            return [os.path.join(self._data_dir, subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([Cifar10TF.NUM_CHANNELS * Cifar10TF.ORIG_IMAGE_HEIGHT * Cifar10TF.ORIG_IMAGE_WIDTH])

        image = tf.cast(
            tf.transpose(tf.reshape(image, [Cifar10TF.NUM_CHANNELS, Cifar10TF.ORIG_IMAGE_HEIGHT, Cifar10TF.ORIG_IMAGE_WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        return image, label

    def preprocess(self, image, label):
        """Preprocess a single image in [height, width, depth] layout."""
        # Pad 4 pixels on each dimension of feature map, done in mini-batch
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       Cifar10TF.ORIG_IMAGE_HEIGHT+Cifar10TF.IMAGE_RAND_CROP_PADDING*2,
                                                       Cifar10TF.ORIG_IMAGE_WIDTH+Cifar10TF.IMAGE_RAND_CROP_PADDING*2)
        image = tf.random_crop(image, [Cifar10TF.IMAGE_HEIGHT, Cifar10TF.IMAGE_WIDTH, Cifar10TF.NUM_CHANNELS])
        image = tf.image.random_flip_left_right(image)

        return image, label

    def make_batch(self, subset, batch_size, distorted):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames(subset)
        dataset = tf.data.TFRecordDataset(filenames)

        if self._shard:
            dataset = dataset.shard(hvd.size(), hvd.rank())

        # Repeat infinitely.
        dataset = dataset.repeat()

        # Parse records.
        dataset = dataset.map(self.parser, num_parallel_calls=self._preproc_threads)

        # Potentially shuffle records.
        if subset == 'train':
            if distorted:
                dataset = dataset.map(self.preprocess, num_parallel_calls=self._preproc_threads)

            min_queue_examples = int(Cifar10TF.TRAIN_NUM_PER_EPOCH * 0.4)
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3*batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=self._preproc_threads)

        # Get output tensors
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    def distorted_inputs(self, distorted=True, eval_data=False):
        if not eval_data:
            subset = 'train'
            batch_size = self._batch_size
        else:
            subset = 'validation'
            batch_size = self._valid_size

        images, labels = self.make_batch(subset, batch_size, distorted=distorted)

        return images, labels

    def init_data(self):
        self.train, self.valid = dt.Opt(), dt.Opt()

        self.train.num_batch = Cifar10TF.TRAIN_NUM_PER_EPOCH // self._batch_size
        self.valid.num_batch = Cifar10TF.EVAL_NUM_PER_EPOCH // self._valid_size

        if self._shard:
            self.train.num_batch = self.train.num_batch // hvd.size()
            self.valid.num_batch = self.valid.num_batch // hvd.size()

        return self

    def generate(self):
        self.train.images, self.train.labels = self.distorted_inputs(distorted=self._distorted, eval_data=False)
        self.valid.images, self.valid.labels = self.distorted_inputs(distorted=False, eval_data=True)

        return self


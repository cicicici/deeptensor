from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tarfile

import deeptensor as dt
import tensorflow as tf


class Cifar10(object):

    ORIG_IMAGE_SIZE = 32
    ORIG_LABEL_SIZE = 1
    NUM_CHANNELS = 3
    IMAGE_SIZE = 32
    # padding to each side for random cropping
    IMAGE_RAND_CROP_PADDING = 3
    NUM_CLASSES = 10
    TRAIN_NUM_PER_EPOCH = 50000
    EVAL_NUM_PER_EPOCH = 10000

    def maybe_download_and_extract(self):
        dest_directory = self._data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)

        filename = self._data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(self._data_url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def read_record(self, filename_queue):
        result = dt.Opt()

        label_bytes = Cifar10.ORIG_LABEL_SIZE  # 2 for CIFAR-100

        result.height = Cifar10.ORIG_IMAGE_SIZE
        result.width = Cifar10.ORIG_IMAGE_SIZE
        result.depth = Cifar10.NUM_CHANNELS
        image_bytes = result.height * result.width * result.depth

        record_bytes = label_bytes + image_bytes

        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        decode_data = tf.decode_raw(value, tf.uint8)

        result.label = tf.cast(tf.strided_slice(decode_data, [0], [label_bytes]), tf.int32)

        depth_major = tf.reshape(tf.strided_slice(decode_data, [label_bytes],
                                                  [label_bytes + image_bytes]),
                                 [result.depth, result.height, result.width])
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return result

    def generate_image_and_label_batch(self, image, label, min_queue_examples, batch_size, shuffle):
        num_preprocess_threads = 4
        if shuffle:
          images, labels = tf.train.shuffle_batch(
              [image, label],
              batch_size=batch_size,
              num_threads=num_preprocess_threads,
              capacity=min_queue_examples + 3 * batch_size,
              min_after_dequeue=min_queue_examples)
        else:
          images, labels = tf.train.batch(
              [image, label],
              batch_size=batch_size,
              num_threads=num_preprocess_threads,
              capacity=min_queue_examples + 3 * batch_size)

        #tf.summary.image('batch-images', images)

        # RANK ADJ
        #return images, tf.reshape(labels, [batch_size])

        return images, labels

    def read_inputs(self, data_dir, batch_size, distorted=True, eval_data=False):

        if not eval_data:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
            num_examples_per_epoch = Cifar10.TRAIN_NUM_PER_EPOCH
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]
            num_examples_per_epoch = Cifar10.EVAL_NUM_PER_EPOCH

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        filename_queue = tf.train.string_input_producer(filenames)

        record = self.read_record(filename_queue)
        casted_image = tf.cast(record.uint8image, tf.float32)

        height = Cifar10.IMAGE_SIZE
        width = Cifar10.IMAGE_SIZE

        if not distorted:
            proc_image = tf.image.resize_image_with_crop_or_pad(casted_image, height, width)
        else:
            proc_image = tf.image.resize_image_with_crop_or_pad(casted_image,
                                                                height+Cifar10.IMAGE_RAND_CROP_PADDING*2,
                                                                width+Cifar10.IMAGE_RAND_CROP_PADDING*2)
            proc_image = tf.random_crop(proc_image, [height, width, Cifar10.NUM_CHANNELS])
            proc_image = tf.image.random_flip_left_right(proc_image)
            #proc_image = tf.image.random_brightness(proc_image, max_delta=63)
            #proc_image = tf.image.random_contrast(proc_image, lower=0.2, upper=1.8)

        float_image = proc_image
        #float_image = tf.image.per_image_standardization(proc_image)
        #float_image.set_shape([height, width, Cifar10.NUM_CHANNELS])

        if  self._out_height != height or self._out_width != width:
            float_image = tf.image.resize_images(float_image,
                                                 [self._out_height, self._out_width],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # RANK ADJ
        #record.label.set_shape([1])
        int_label = tf.reshape(record.label, [])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        return self.generate_image_and_label_batch(float_image, int_label,
                                                   min_queue_examples, batch_size,
                                                   shuffle=True)

    def distorted_inputs(self, batch_size, distorted=True, eval_data=False, use_fp16=False):
        data_dir = os.path.join(self._data_dir, 'cifar-10-batches-bin')
        images, labels = self.read_inputs(data_dir=data_dir,
                                          batch_size=batch_size,
                                          distorted=distorted,
                                          eval_data=eval_data)
        if use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)

        return images, labels

    def __init__(self, batch_size=128, valid_size=128, reshape=False, one_hot=False, distorted=False,
                 out_height=IMAGE_SIZE, out_width=IMAGE_SIZE):

        self._data_dir = './_asset/data/cifar10'
        self._data_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

        self._batch_size = batch_size
        self._valid_size = valid_size
        self._reshape = reshape
        self._one_hot = one_hot
        self._distorted = distorted
        self._out_height = out_height
        self._out_width = out_width

    def init_data(self):
        self.train, self.valid = dt.Opt(), dt.Opt()

        self.maybe_download_and_extract()

        self.train.num_batch = Cifar10.TRAIN_NUM_PER_EPOCH // self._batch_size
        self.valid.num_batch = Cifar10.EVAL_NUM_PER_EPOCH // self._valid_size

        return self

    def generate(self):
        self.train.images, self.train.labels = self.distorted_inputs(self._batch_size, distorted=self._distorted, eval_data=False)
        self.valid.images, self.valid.labels = self.distorted_inputs(self._valid_size, distorted=False, eval_data=True)

        return self


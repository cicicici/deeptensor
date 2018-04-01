from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def _data_to_tensor_single(data_list, batch_size, name=None):
    _num_thread = 4
    _cap_thread = _num_thread
    _cap_max = 128
    _cap_min = 16

    const_list = [tf.constant(data) for data in data_list]
    queue_list = tf.train.slice_input_producer(const_list,
                                               capacity = batch_size * _cap_max * _cap_thread,
                                               name = name, shuffle = False)

    return tf.train.shuffle_batch(queue_list, batch_size,
                                  capacity = batch_size * _cap_max * _cap_thread,
                                  min_after_dequeue = batch_size * _cap_min * _cap_thread,
                                  name = name, num_threads = _num_thread)

def _data_to_tensor(data_list, batch_size, name=None):
    _num_thread = 4
    _cap_thread = _num_thread
    _cap_max = 128
    _cap_min = 16

    const_list = [tf.constant(data) for data in data_list]

    images, labels = tf.train.shuffle_batch(const_list, batch_size,
                                            capacity = batch_size * _cap_max * _cap_thread,
                                            min_after_dequeue = batch_size * _cap_min * _cap_thread,
                                            name = name, num_threads = _num_thread,
                                            enqueue_many=True)

    tf.summary.image('batch-images', images)

    return images, labels

class Mnist(object):

    _data_dir = './_asset/data/mnist'

    ORIG_IMAGE_SIZE = 28
    ORIG_LABEL_SIZE = 1
    NUM_CHANNELS = 1
    IMAGE_SIZE = 28
    NUM_CLASSES = 10
    TRAIN_NUM_PER_EPOCH = 50000
    EVAL_NUM_PER_EPOCH = 10000

    def __init__(self, batch_size=128, valid_size=128, reshape=False, one_hot=False):
        self._data_dir = Mnist._data_dir
        self._batch_size = batch_size
        self._valid_size = valid_size
        self._reshape = reshape
        self._one_hot = one_hot

    def init_data(self):
        self.train, self.valid, self.test = dt.Opt(), dt.Opt, dt.Opt()

        self.train.num_batch = Mnist.TRAIN_NUM_PER_EPOCH // self._batch_size
        self.valid.num_batch = Mnist.EVAL_NUM_PER_EPOCH // self._valid_size

        return self

    def generate(self):
        self._data_set = input_data.read_data_sets(Mnist._data_dir, reshape=self._reshape, one_hot=self._one_hot)

        self._train_raw = self._data_set.train
        self._valid_raw = self._data_set.validation
        self._test_raw = self._data_set.test

        self.train.images, self.train.labels = \
            _data_to_tensor([self._train_raw.images, self._train_raw.labels.astype('int32')], self._batch_size, name='train')
        self.valid.images, self.valid.labels = \
            _data_to_tensor([self._valid_raw.images, self._valid_raw.labels.astype('int32')], self._valid_size, name='valid')

        return self


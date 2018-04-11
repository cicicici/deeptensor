from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf
import horovod.tensorflow as hvd


class ImageTFRecord(object):

    def __init__(self, filenames, num_images, num_classes, out_height, out_width, channels,
                 batch_size=32, shuffle=True, shuffle_size=0, epochs=10000, shard=True,
                 is_training=True, distorted=True, one_hot=False,
                 preproc_threads=4, record_prefetch_factor=4, batch_prefetch_factor=0.5):

        self._filenames = filenames
        self._num_files = len(filenames)
        self._num_images = num_images
        self._num_classes = num_classes
        self._out_height = out_height
        self._out_width = out_width
        self._channels = channels

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._shuffle_size = shuffle_size
        self._epochs = epochs

        self._shard = shard
        if self._shard:
            self._num_images = num_images // hvd.size()

        self._is_training = is_training
        self._distorted = distorted
        self._one_hot = one_hot

        self._preproc_threads = preproc_threads
        self._record_prefetch_factor = record_prefetch_factor
        self._batch_prefetch_factor = batch_prefetch_factor

    def parse_example_proto(self, example_serialized):
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                    default_value=-1)
        }

        features = tf.parse_single_example(example_serialized, feature_map)

        return features['image/encoded'], features['image/class/label']

    def parse_record(self, raw_record, is_training):
        image, label = self.parse_example_proto(raw_record)

        image = tf.image.decode_jpeg(image, channels=self._channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        image = dt.data.vgg_preprocessing.preprocess_image(image, self._out_height, self._out_width, is_training=is_training)
        label = tf.cast(tf.reshape(label, shape=[]), dtype=tf.int32)

        if self._one_hot:
            label = tf.one_hot(label, self._num_classes)

        #if is_training:
        #    image, label = dt.data.image_processing.distort_image(image, label, height=self._out_height, width=self._out_width, resize=False, flip=False, color=True)

        return image, label

    def init_data(self):
        self._num_batch = self._num_images // self._batch_size
        return self

    def generate(self):
        dataset = tf.data.Dataset.from_tensor_slices(self._filenames)
        if self._shard:
            dataset = dataset.shard(hvd.size(), hvd.rank())
        dataset = dataset.repeat(self._epochs)

        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=self._num_files)
        dataset = dataset.repeat()

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=int(self._batch_size * self._record_prefetch_factor))

        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=self._shuffle_size)
        dataset = dataset.repeat()

        dataset = dataset.map(lambda value: self.parse_record(value, self._is_training),
                              num_parallel_calls=self._preproc_threads)
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(buffer_size=int(self._batch_size * self._batch_prefetch_factor))

        dataset_iterator = dataset.make_one_shot_iterator()
        self._images_batch, self._labels_batch = dataset_iterator.get_next()

        return self


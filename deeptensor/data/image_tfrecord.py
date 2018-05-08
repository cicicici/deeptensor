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
                 preproc_threads=4, record_prefetch_factor=4, batch_prefetch_factor=0.5,
                 data_format=dt.dformat.DEFAULT, src_data_format=dt.dformat.NHWC):

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

        self._data_format = data_format
        self._src_data_format = src_data_format

    def _parse_example_proto(self, example_serialized):
        """Parses an Example proto containing a training example of an image.

        The output of the build_image_data.py image preprocessing script is a dataset
        containing serialized Example protocol buffers. Each Example proto contains
        the following fields (values are included as examples):

            image/height: 462
            image/width: 581
            image/colorspace: 'RGB'
            image/channels: 3
            image/class/label: 615
            image/class/synset: 'n03623198'
            image/class/text: 'knee pad'
            image/object/bbox/xmin: 0.1
            image/object/bbox/xmax: 0.9
            image/object/bbox/ymin: 0.2
            image/object/bbox/ymax: 0.6
            image/object/bbox/label: 615
            image/format: 'JPEG'
            image/filename: 'ILSVRC2012_val_00041207.JPEG'
            image/encoded: <JPEG encoded string>

        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
                Example protocol buffer.

        Returns:
            image_buffer: Tensor tf.string containing the contents of a JPEG file.
            label: Tensor tf.int32 containing the label.
            bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
                where each coordinate is [0, 1) and the coordinates are arranged as
                [ymin, xmin, ymax, xmax].
        """
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                    default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                   default_value=''),
        }
        sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        feature_map.update(
            {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                         'image/object/bbox/ymin',
                                         'image/object/bbox/xmax',
                                         'image/object/bbox/ymax']})

        features = tf.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

        # Note that we impose an ordering of (y, x) just to make life difficult.
        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

        # Force the variable number of bounding boxes into the shape
        # [1, num_boxes, coords].
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])

        return features['image/encoded'], label, bbox

    def parse_record(self, raw_record, is_training):
        image_buffer, label, bbox = self._parse_example_proto(raw_record)

        # vgg preprocessing
        #image = tf.image.decode_jpeg(image_buffer, channels=self._channels)
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #image = dt.data.vgg_preprocessing.preprocess_image(image, self._out_height, self._out_width,
        #                                                   is_training=is_training)

        # google preprocessing
        image = dt.data.imagenet_preprocessing.preprocess_image(image_buffer=image_buffer,
                                                                bbox=bbox,
                                                                output_height=self._out_height,
                                                                output_width=self._out_width,
                                                                num_channels=self._channels,
                                                                is_training=is_training)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

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
        #dataset = dataset.repeat(self._epochs)

        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=self._num_files)
        #dataset = dataset.repeat()

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

        self._images_batch = dt.dformat_chk_conv_images(self._images_batch, self._src_data_format, self._data_format)

        return self


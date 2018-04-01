from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


def parse_tfrecord_files(example_serialized):
    feature_map = {
        'image_raw': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image_label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1)
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image_label'], dtype=tf.int32)

    return features['image_raw'], label

def decode_serialized_image(image_buffer, label):

    parsed_image = tf.image.decode_jpeg(image_buffer, channels=3)
    parsed_image = tf.image.convert_image_dtype(parsed_image, dtype=tf.float32)
    distorted_image = tf.image.resize_images(parsed_image, [96, 96])

    return distorted_image, label

def parse_filename_example_png(image_filename, label):

    image_buffer = tf.read_file(image_filename)
    parsed_image = tf.image.decode_png(image_buffer, channels=3)
    parsed_image = tf.image.convert_image_dtype(parsed_image, dtype=tf.float32)

    return parsed_image, label

def parse_filename_example_jpeg(image_filename, label):

    image_buffer = tf.read_file(image_filename)
    parsed_image = tf.image.decode_jpeg(image_buffer, channels=3)
    parsed_image = tf.image.convert_image_dtype(parsed_image, dtype=tf.float32)

    return parsed_image, label

def parse_raw_file(filename, label):

    string_buffer = tf.read_file(filename)
    parsed_data = tf.decode_raw(string_buffer, tf.float32)
    parsed_data = tf.reshape(parsed_data, [511, 120])

    return parsed_data, label

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  TODO(coreylynch): add as a dependency, when slim or tensorflow/models are
  pipfied.
  Source:
  https://raw.githubusercontent.com/tensorflow/models/a9d0e6e8923a4/slim/preprocessing/inception_preprocessing.py

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather than adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def distort_image(image, label, height=96, width=96, resize=True, flip=True, color=True):
    out_image = image

    if resize:
        out_image = tf.image.resize_images(out_image, [height, width])
        out_image.set_shape([height, width, 3])

    if flip:
        # Randomly flip the image horizontally.
        out_image = tf.image.random_flip_left_right(out_image)

    if color:
        # Randomly distort the colors.
        out_image = distort_color(out_image, color_ordering=0, fast_mode=True, scope=None)

    return out_image, label


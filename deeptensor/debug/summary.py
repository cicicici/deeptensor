from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf

from tensorflow.python.ops import gen_logging_ops


def tensor_name(tensor):
    return ''.join(tensor.name.split(':')[:-1])

def tensor_short_name(tensor):
    return ''.join(tensor.name.split(':')[:-1]).split('/')[-1]

def _scalar(name, tensor, skip_reuse=False):
    if skip_reuse or (not tf.get_variable_scope().reuse and not dt.get_ctx().reuse):
        val = gen_logging_ops.scalar_summary(name, tensor)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, val)

def _histogram(name, tensor, skip_reuse=False):
    if skip_reuse or (not tf.get_variable_scope().reuse and not dt.get_ctx().reuse):
        val = gen_logging_ops.histogram_summary(name, tensor)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, val)

def summary_loss(tensor, prefix='losses', name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + tensor_name(tensor) if name is None else prefix + name
    # summary statistics
    _scalar(name, tf.reduce_mean(tensor), skip_reuse=True)
    _histogram(name + '-h', tensor, skip_reuse=True)

def summary_metric(tensor, prefix='metrics', name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + tensor_name(tensor) if name is None else prefix + name
    # summary statistics
    _scalar(name, tf.reduce_mean(tensor))
    _histogram(name + '-h', tensor)

def summary_gradient(tensor, gradient, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + tensor_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    _scalar(name + '/grad', tf.reduce_mean(tf.abs(gradient)))
    _histogram(name + '/grad-h', tf.abs(gradient))

def summary_activation(tensor, prefix='act', name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + tensor_name(tensor) if name is None else prefix + name
    # summary statistics
    _scalar(name + '/ratio',
            tf.reduce_mean(tf.cast(tf.greater(tensor, 0), dt.floatx)))
    _histogram(name + '/ratio-h', tensor)

def summary_param(tensor, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + tensor_name(tensor) if name is None else prefix + name
    # summary statistics
    _scalar(name + '/abs', tf.reduce_mean(tf.abs(tensor)))
    _histogram(name + '/abs-h', tf.abs(tensor))

def summary_image(tensor, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + tensor_name(tensor) if name is None else prefix + name
    # summary statistics
    if not tf.get_variable_scope().reuse:
        tf.summary.image(name + '-im', tensor)

def summary_audio(tensor, sample_rate=16000, prefix=None, name=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + tensor_name(tensor) if name is None else prefix + name
    # summary statistics
    if not tf.get_variable_scope().reuse:
        tf.summary.audio(name + '-au', tensor, sample_rate)


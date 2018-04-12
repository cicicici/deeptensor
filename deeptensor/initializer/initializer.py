from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf

import numpy as np


def constant(name, shape, value=0, dtype=dt.floatx, summary=True, regularizer=None, trainable=True):
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.constant_initializer(value),
                        regularizer=regularizer, trainable=trainable)
    # add summary
    if summary:
        dt.summary_param(x)
    return x

def uniform(name, shape, scale=0.05, dtype=dt.floatx, summary=True, regularizer=None, trainable=True):
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale),
                        regularizer=regularizer, trainable=trainable)
    # add summary
    if summary:
        dt.summary_param(x)
    return x

def he_uniform(name, shape, scale=1, dtype=dt.floatx, summary=True, regularizer=None, trainable=True):
    fin, _ = _get_fans(shape)
    s = np.sqrt(1. * scale / fin)
    return uniform(name, shape, s, dtype, summary, regularizer, trainable)

def glorot_uniform(name, shape, scale=1, dtype=dt.floatx, summary=True, regularizer=None, trainable=True):
    fin, fout = _get_fans(shape)
    s = np.sqrt(6. * scale / (fin + fout))
    return uniform(name, shape, s, dtype, summary, regularizer, trainable)

def variance_scaling(name, shape, scale=1, mode='fan_in', distribution='normal', dtype=dt.floatx, summary=True, regularizer=None, trainable=True):
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.variance_scaling_initializer(scale=scale,
                                                                    mode=mode,
                                                                    distribution=distribution,
                                                                    dtype=dtype),
                        regularizer=regularizer, trainable=trainable)
    # add summary
    if summary:
        dt.summary_param(x)
    return x

def identity(name, dim, scale=1, dtype=dt.floatx, summary=True, regularizer=None, trainable=True):
    x = tf.get_variable(name,
                        initializer=tf.constant(np.eye(dim) * scale, dtype=dtype),
                        regularizer=regularizer, trainable=trainable)
    # add summary
    if summary:
        dt.summary_param(x)
    return x

def orthogonal(name, shape, scale=1.1, dtype=dt.floatx, summary=True, regularizer=None, trainable=True):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    # create variable
    x = tf.get_variable(name,
                        initializer=tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype),
                        regularizer=regularizer, trainable=trainable)
    # add summary
    if summary:
        dt.summary_param(x)
    return x

def external(name, value, dtype=dt.floatx, summary=True, regularizer=None, trainable=True):
    # create variable
    x = tf.get_variable(name,
                        initializer=tf.constant(value, dtype=dtype),
                        regularizer=regularizer, trainable=trainable)
    # add summary
    if summary:
        dt.summary_param(x)
    return x

def _get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        kernel_size = np.prod(shape[:2])
        fan_in = shape[-2] * kernel_size
        fan_out = shape[-1] * kernel_size
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


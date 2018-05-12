from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deeptensor as dt
import tensorflow as tf


@dt.dec_sugar_func
def pool(tensor, opt):
    # default stride and pad
    opt += dt.Opt(stride=(2, 2), pad='VALID', dout=0)
    # shape size
    opt += dt.Opt(size=opt.stride, padding=None)

    in_shape = tensor.get_shape().as_list()
    in_dim = dt.tensor.get_dim(tensor, opt.data_format)
    opt += dt.Opt(shape=in_shape, in_dim=in_dim)

    # default to average pooling
    opt += dt.Opt(avg=True)

    size = dt.tensor.get_size(opt.size, opt.data_format)
    stride = dt.tensor.get_stride(opt.stride, opt.data_format)
    padding = dt.tensor.get_padding(opt.padding, opt.data_format) if opt.padding else None
    #dt.log_pp(dt.DC.NET, dt.DL.DEBUG, opt)
    dt.debug(dt.DC.NET, "\t[TRANS] {}, T {}, R {}, shape {}, in_dim {}, size {}, stride {}"
                             .format(opt.name, opt.is_training, opt.reuse, opt.shape, opt.in_dim, size, stride))
    dt.debug(dt.DC.NET, "\t            pad {}, padding {}, avg {}, dout {}, df {}"
                             .format(opt.pad, padding, opt.avg, opt.dout, opt.data_format))
    dt.dformat_chk(opt.data_format)

    with tf.variable_scope(opt.name, reuse=opt.reuse) as scope:
        if padding is not None:
            tensor_in = tf.pad(tensor,
                               paddings=tf.constant(padding, dtype=tf.float32),
                               mode="CONSTANT",
                               constant_values=0)
        else:
            tensor_in = tensor

        if opt.avg:
            out = tf.nn.avg_pool(tensor_in, size, stride, opt.pad, data_format=opt.data_format)
        else:
            out = tf.nn.max_pool(tensor_in, size, stride, opt.pad, data_format=opt.data_format)

        # apply dropout
        if opt.is_training and opt.dout and (opt.dout > 0 and opt.dout < 1):
            out = tf.nn.dropout(out, 1 - opt.dout),

    return out

@dt.dec_sugar_func
def global_pool(tensor, opt):
    in_shape = tensor.get_shape().as_list()
    in_dim = dt.tensor.get_dim(tensor, opt.data_format)
    opt += dt.Opt(shape=in_shape, in_dim=in_dim)

    # default to average pooling
    opt += dt.Opt(avg=True)

    #dt.log_pp(dt.DC.NET, dt.DL.DEBUG, opt)
    dt.debug(dt.DC.NET, "\t[TRANS] {}, T {}, R {}, shape {}, in_dim {}, avg {}, dout {}, df {}"
                             .format(opt.name, opt.is_training, opt.reuse, opt.shape, opt.in_dim,
                                     opt.avg, opt.dout, opt.data_format))
    dt.dformat_chk(opt.data_format)

    with tf.variable_scope(opt.name, reuse=opt.reuse) as scope:
        if opt.data_format == dt.dformat.NHWC:
            axis = [1, 2]
        elif opt.data_format == dt.dformat.NCHW:
            axis = [2, 3]

        if opt.avg:
            out = tf.reduce_mean(tensor, axis)
        else:
            out = tf.reduce_max(tensor, axis)

        # apply dropout
        if opt.is_training and opt.dout and (opt.dout > 0 and opt.dout < 1):
            out = tf.nn.dropout(out, 1 - opt.dout),

    return out

@dt.dec_sugar_func
def flatten(tensor, opt):
    shape = tensor.get_shape().as_list()
    dim = np.prod(shape[1:])
    dt.debug(dt.DC.NET, "\t[TRANS] {}, shape {}, out {}"
                             .format(opt.name, shape, dim))

    with tf.variable_scope(opt.name, reuse=opt.reuse) as scope:
        out = tf.reshape(tensor, [-1, dim])

    return out

@dt.dec_sugar_func
def argmax(tensor, opt):
    opt += dt.Opt(axis=tensor.get_shape().ndims-1)
    return tf.argmax(tensor, opt.axis, opt.name)

@dt.dec_sugar_func
def float(tensor, opt):
    return tf.cast(tensor, dt.floatx, name=opt.name)


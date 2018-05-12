from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf

from deeptensor.layer import layer_ctx as layer_ctx


@layer_ctx.dec_layer_func
def conv(tensor, opt):
    # default options
    opt += dt.Opt(size=(3, 3), stride=(1, 1), pad='VALID', padding=None)

    size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]

    stride = dt.tensor.get_stride(opt.stride, opt.data_format)
    padding = dt.tensor.get_padding(opt.padding, opt.data_format) if opt.padding else None

    #dt.log_pp(dt.DC.NET, dt.DL.DEBUG, opt)
    dt.debug(dt.DC.NET, "\t\t            [conv] size {}, in {}, out {}, stride {}, pad {}, padding {}, bias {}, filler {}"
                             .format(size, opt.in_dim, opt.dim, stride, opt.pad, padding, opt.bias, opt.weight_filler))

    if padding is not None:
        tensor_in = tf.pad(tensor,
                           paddings=tf.constant(padding, dtype=tf.float32),
                           mode="CONSTANT",
                           constant_values=0)
    else:
        tensor_in = tensor

    # parameter initialize
    if opt.weight_filler == 'xavier':
        w = dt.initializer.glorot_uniform('W', (size[0], size[1], opt.in_dim, opt.dim),
                                          regularizer=opt.regularizer_func, summary=opt.summary)
    elif opt.weight_filler == 'he':
        w = dt.initializer.he_uniform('W', (size[0], size[1], opt.in_dim, opt.dim),
                                      regularizer=opt.regularizer_func, summary=opt.summary)
    else:
        w = dt.initializer.variance_scaling('W', (size[0], size[1], opt.in_dim, opt.dim),
                                            scale=2.0, mode='fan_out',
                                            regularizer=opt.regularizer_func, summary=opt.summary)

    # apply convolution
    out = tf.nn.conv2d(tensor_in, w, strides=stride, padding=opt.pad, data_format=opt.data_format)

    if opt.bias:
        b = dt.initializer.constant('b', opt.dim, summary=opt.summary)
        out = tf.nn.bias_add(out, b, data_format=opt.data_format)

    return out

@layer_ctx.dec_layer_func
def dense(tensor, opt):
    #dt.log_pp(dt.DC.NET, dt.DL.DEBUG, opt)
    dt.debug(dt.DC.NET, "\t\t            [dense] in {}, out {}, bias {}, filler {}"
                             .format(opt.in_dim, opt.dim, opt.bias, opt.weight_filler))
    # parameter initialize
    if opt.weight_filler == 'xavier':
        w = dt.initializer.glorot_uniform('W', (opt.in_dim, opt.dim),
                                          regularizer=opt.regularizer_func, summary=opt.summary)
    elif opt.weight_filler == 'he':
        w = dt.initializer.he_uniform('W', (opt.in_dim, opt.dim),
                                      regularizer=opt.regularizer_func, summary=opt.summary)
    else:
        w = dt.initializer.variance_scaling('W', (opt.in_dim, opt.dim),
                                            scale=2.0, mode='fan_out',
                                            regularizer=opt.regularizer_func, summary=opt.summary)

    # apply transform
    out = tf.matmul(tensor, w)

    if opt.bias:
        b = dt.initializer.constant('b', opt.dim, summary=opt.summary)
        out = tf.nn.bias_add(out, b)

    return out

@layer_ctx.dec_layer_func
def bypass(tensor, opt):
    dt.debug(dt.DC.NET, "\t\t            [bypass] in {}, out {}"
                             .format(opt.in_dim, opt.dim))
    return tensor


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf

from deeptensor.layer import layer_ctx as layer_ctx


@layer_ctx.dec_layer_func
def conv(tensor, opt):
    # default options
    opt += dt.Opt(size=(3, 3), stride=(1, 1, 1, 1), pad='VALID', padding=None)

    size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]

    stride = dt.utils.get_stride(opt.stride)
    padding = dt.utils.get_padding(opt.padding) if opt.padding else None
    if padding is not None:
        tensor_in = tf.pad(tensor,
                           paddings=tf.constant(padding, dtype=tf.int32),
                           mode="CONSTANT",
                           constant_values=0)
    else:
        tensor_in = tensor

    #dt.log_pp(dt.DC.NET, dt.DL.DEBUG, opt)
    dt.debug(dt.DC.NET, "            {}, size {}, in {}, out {}, stride {}, pad {}, padding {}, bias {}, filler {}"
                             .format(opt.name, size, opt.in_dim, opt.dim, stride, opt.pad, padding, opt.bias, opt.weight_filler))

    # parameter initialize
    if opt.weight_filler == 'xavier':
        w = dt.initializer.glorot_uniform('W', (size[0], size[1], opt.in_dim, opt.dim),
                                          regularizer=opt.regularizer_func, summary=opt.summary)
    elif opt.weight_filler == 'he':
        w = dt.initializer.he_uniform('W', (size[0], size[1], opt.in_dim, opt.dim),
                                      regularizer=opt.regularizer_func, summary=opt.summary)
    else:
        w = dt.initializer.variance_scaling('W', (size[0], size[1], opt.in_dim, opt.dim),
                                            regularizer=opt.regularizer_func, summary=opt.summary)

    b = dt.initializer.constant('b', opt.dim, summary=opt.summary) if opt.bias else 0

    # apply convolution
    out = tf.nn.conv2d(tensor_in, w, strides=stride, padding=opt.pad) + b

    return out

@layer_ctx.dec_layer_func
def dense(tensor, opt):
    #dt.log_pp(dt.DC.NET, dt.DL.DEBUG, opt)
    dt.debug(dt.DC.NET, "            {}, in {}, out {}, bias {}, filler {}"
                             .format(opt.name, opt.in_dim, opt.dim, opt.bias, opt.weight_filler))
    # parameter initialize
    if opt.weight_filler == 'xavier':
        w = dt.initializer.glorot_uniform('W', (opt.in_dim, opt.dim),
                                          regularizer=opt.regularizer_func, summary=opt.summary)
    else:
        w = dt.initializer.he_uniform('W', (opt.in_dim, opt.dim),
                                      regularizer=opt.regularizer_func, summary=opt.summary)
    b = dt.initializer.constant('b', opt.dim, summary=opt.summary) if opt.bias else 0

    # apply transform
    out = tf.matmul(tensor, w) + b

    return out

@layer_ctx.dec_layer_func
def bypass(tensor, opt):
    dt.debug(dt.DC.NET, "            {}, in {}, out {}, bias {}, filler {}"
                             .format(opt.name, opt.in_dim, opt.dim, opt.bias, opt.weight_filler))
    return tensor


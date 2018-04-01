from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import deeptensor as dt
import tensorflow as tf


@dt.dec_sugar_func
def pool(tensor, opt):

    # default stride and pad
    opt += dt.Opt(stride=(1, 2, 2, 1), pad='VALID', dout=0)

    # shape size
    opt += dt.Opt(size=opt.stride, padding=None)

    shape = tensor.get_shape().as_list()
    opt += dt.Opt(shape=shape, in_dim=shape[-1])

    size = opt.size if isinstance(opt.size, (list, tuple)) else [1, opt.size, opt.size, 1]
    size = [1, size[0], size[1], 1] if len(size) == 2 else size

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
    dt.debug(dt.DC.NET, "[TRANS] {}, T {}, R {}, shape {}, size {}, stride {}, pad {}, padding {}, avg {}, dout {}"
                             .format(opt.name, opt.is_training, opt.reuse, opt.shape, size, stride, opt.pad, padding, opt.avg, opt.dout))

    if opt.avg:
        out = tf.nn.avg_pool(tensor_in, size, stride, opt.pad)
    else:
        out = tf.nn.max_pool(tensor_in, size, stride, opt.pad)

    # apply dropout
    if opt.is_training and opt.dout and (opt.dout > 0 and opt.dout < 1):
        out = tf.nn.dropout(out, 1 - opt.dout),

    return tf.identity(out, name=opt.name)

@dt.dec_sugar_func
def flatten(tensor, opt):
    shape = tensor.get_shape().as_list()
    dim = np.prod(shape[1:])
    #dt.log_pp(dt.DC.NET, dt.DL.DEBUG, opt)
    dt.debug(dt.DC.NET, "[TRANS] flatten, shape {}, out {}"
                             .format(shape, dim))

    return tf.reshape(tensor, [-1, dim], name=opt.name)

@dt.dec_sugar_func
def argmax(tensor, opt):
    opt += dt.Opt(axis=tensor.get_shape().ndims-1)
    return tf.argmax(tensor, opt.axis, opt.name)

@dt.dec_sugar_func
def float(tensor, opt):
    return tf.cast(tensor, dt.floatx, name=opt.name)


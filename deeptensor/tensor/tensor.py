from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import deeptensor as dt
import tensorflow as tf


def get_stride(stride):
    if len(stride) == 1:
        _stride = [1, stride, stride, 1]
    elif len(stride) == 2:
        _stride = [1, stride[0], stride[1], 1]
    elif len(stride) == 4:
        _stride = stride
    else:
        raise ValueError('The rank of stride has to be 1, 2, or 4')

    return _stride

def get_padding(padding):
    if not padding:
        pad_h, pad_w = 0, 0
    elif len(padding) == 1:
        pad_h = padding[0]
        pad_w = padding[0]
    elif len(padding) == 2:
        pad_h, pad_w = padding[0], padding[1]
    else:
        raise ValueError('The padding format is incorrect')

    return [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]

def get_dim(tensor, data_format=dt.dformat.DEFAULT):
    shape = tensor.get_shape().as_list()
    if data_format == dt.dformat.NHWC:
        dim = shape[-1]
    elif data_format == dt.dformat.NCHW:
        dim = shape[1]
    else:
        raise ValueError('Invalid data format')
    return dim


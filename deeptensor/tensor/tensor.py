from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import deeptensor as dt
import tensorflow as tf


def get_size(size, data_format=dt.dformat.DEFAULT):
    if len(size) == 1:
        if data_format == dt.dformat.NHWC:
            _size = [1, size, size, 1]
        elif data_format == dt.dformat.NCHW:
            _size = [1, 1, size, size]
    elif len(size) == 2:
        if data_format == dt.dformat.NHWC:
            _size = [1, size[0], size[1], 1]
        elif data_format == dt.dformat.NCHW:
            _size = [1, 1, size[0], size[1]]
    elif len(size) == 4:
        _size = size
    else:
        raise ValueError('The rank of size has to be 1, 2, or 4')

    return _size

def get_stride(stride, data_format=dt.dformat.DEFAULT):
    if len(stride) == 1:
        if data_format == dt.dformat.NHWC:
            _stride = [1, stride, stride, 1]
        elif data_format == dt.dformat.NCHW:
            _stride = [1, 1, stride, stride]
    elif len(stride) == 2:
        if data_format == dt.dformat.NHWC:
            _stride = [1, stride[0], stride[1], 1]
        elif data_format == dt.dformat.NCHW:
            _stride = [1, 1, stride[0], stride[1]]
    elif len(stride) == 4:
        _stride = stride
    else:
        raise ValueError('The rank of stride has to be 1, 2, or 4')

    return _stride

def get_padding(padding, data_format=dt.dformat.DEFAULT):
    if not padding:
        pad_h, pad_w = 0, 0
    elif len(padding) == 1:
        pad_h = padding[0]
        pad_w = padding[0]
    elif len(padding) == 2:
        pad_h, pad_w = padding[0], padding[1]
    else:
        raise ValueError('The padding format is incorrect')

    if data_format == dt.dformat.NHWC:
        paddings = [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]]
    elif data_format == dt.dformat.NCHW:
        paddings = [[0, 0], [0, 0], [pad_h, pad_h], [pad_w, pad_w]]

    return paddings

def get_dim(tensor, data_format=dt.dformat.DEFAULT):
    shape = tensor.get_shape().as_list()
    if data_format == dt.dformat.NHWC:
        dim = shape[-1]
    elif data_format == dt.dformat.NCHW:
        dim = shape[1]
    else:
        raise ValueError('Invalid data format')
    return dim


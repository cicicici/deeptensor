from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import deeptensor as dt
import tensorflow as tf


def get_size(size, data_format):
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

def get_stride(stride, data_format):
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

def get_padding(padding, data_format):
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

def get_padding_channel(padding, data_format):
    if not padding:
        pad_l, pad_r = 0, 0
    elif len(padding) == 1:
        pad_l = padding[0]
        pad_r = padding[0]
    elif len(padding) == 2:
        pad_l, pad_r = padding[0], padding[1]
    else:
        raise ValueError('The padding format is incorrect')

    if data_format == dt.dformat.NHWC:
        paddings = [[0, 0], [0, 0], [0, 0], [pad_l, pad_r]]
    elif data_format == dt.dformat.NCHW:
        paddings = [[0, 0], [pad_l, pad_r], [0, 0], [0, 0]]

    return paddings

def get_shape(tensor):
    return tensor.get_shape().as_list()

def get_channel_axis(tensor, data_format):
    shape = tensor.get_shape()
    if shape.ndims == 2:
        axis = 1
    elif shape.ndims == 4:
        if data_format == dt.dformat.NHWC:
            axis = 3
        elif data_format == dt.dformat.NCHW:
            axis = 1
        else:
            raise ValueError('Invalid data format')
    else:
        raise ValueError('Invalid tensor dimentions')

    return axis

def get_dim(tensor, data_format):
    shape = get_shape(tensor)
    axis = get_channel_axis(tensor, data_format)
    dim = shape[axis]
    return dim


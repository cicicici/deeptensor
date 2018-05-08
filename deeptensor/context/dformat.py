from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


class dformat(object):
    NHWC    = "NHWC"
    NCHW    = "NCHW"
    DEFAULT = NHWC

def dformat_to_tl(df):
    df_tl = None
    if df == dformat.NHWC:
        df_tl = "channels_last"
    elif df == dformat.NCHW:
        df_tl = "channels_first"
    return df_tl

def dformat_chk(fmt):
    if fmt != dformat.NHWC and fmt != dformat.NCHW:
        raise ValueError('Invalid data format [{}]'.format(fmt))

def dformat_chk_conv_image(image, in_fmt, out_fmt):
    dformat_chk(in_fmt)
    dformat_chk(out_fmt)
    if in_fmt == dformat.NHWC and out_fmt == dformat.NCHW:
        image = tf.transpose(image, [2, 0, 1])
    elif in_fmt == dformat.NCHW and out_fmt == dformat.NHWC:
        image = tf.transpose(image, [1, 2, 0])
    return image

def dformat_chk_conv_images(images, in_fmt, out_fmt):
    dformat_chk(in_fmt)
    dformat_chk(out_fmt)
    if in_fmt == dformat.NHWC and out_fmt == dformat.NCHW:
        images = tf.transpose(images, [0, 3, 1, 2])
    elif in_fmt == dformat.NCHW and out_fmt == dformat.NHWC:
        images = tf.transpose(images, [0, 2, 3, 1])
    return images


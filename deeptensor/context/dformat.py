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


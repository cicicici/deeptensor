from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


# ratio = r // stages, so default is 16 // 4 = 4 for resnet
def se_block(tensor, dim, out_dim, data_format, se_ratio, name='se_block'):
    with dt.ctx(name=name):
        n = dt.transform.global_pool(tensor, avg=True, name='global_avg')
        n = dt.layer.dense(n, dim=dim // se_ratio, act='relu', bn=False, ln=False, name='fc1')
        n = dt.layer.dense(n, dim=out_dim, act='sigmoid', bn=False, ln=False, name='fc2')
        chn_axis = dt.tensor.get_channel_axis(tensor, data_format)
        shape = [-1, 1, 1, 1]
        shape[chn_axis] = out_dim
        n = tf.reshape(n, shape)
    return n


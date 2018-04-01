from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


def lenet5_a(in_tensor, num_classes):

    with dt.ctx(act='relu', bn=True):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = in_tensor
        n = dt.layer.conv(n, dim=16)
        n = dt.transform.pool(n)
        n = dt.layer.conv(n, dim=32)
        n = dt.transform.pool(n)
        n = dt.layer.conv(n, dim=32)
        n = dt.transform.pool(n)
        n = dt.transform.flatten(n)
        n = dt.layer.dense(n, dim=256)
        n = dt.layer.dense(n, dim=num_classes, act='linear', bn=False)

    return n

def lenet5(in_tensor, num_classes):

    # conv layers
    with dt.ctx(name='convs', act='relu', bn=True):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = in_tensor
        n = dt.layer.conv(n, dim=16, name='conv1')
        n = dt.transform.pool(n)
        n = dt.layer.conv(n, dim=32, name='conv2')
        n = dt.transform.pool(n)
        n = dt.layer.conv(n, dim=32, name='conv3')
        n = dt.transform.pool(n)

    # fc layers
    with dt.ctx(name='fcs', act='relu', bn=True):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = dt.transform.flatten(n)
        n = dt.layer.dense(n, dim=256, name='fc1')
        n = dt.layer.dense(n, dim=num_classes, act='linear', bn=False, name='fc2')

    return n


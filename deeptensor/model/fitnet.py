from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


def fitnet4(in_tensor, num_classes):

    # conv layers
    with dt.ctx(name='convs', act='relu', bn=True, weight_filler='xavier'):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())

        regularizer='l2'
        conv_decay=5e-07
        fc_decay=1e-06

        n = in_tensor
        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv11')
        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv12')
        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv13')
        n = dt.layer.conv(n, size=(3, 3), dim=48, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv14')
        n = dt.layer.conv(n, size=(3, 3), dim=48, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv15')
        n = dt.transform.pool(n, size=(2, 2), stride=[2, 2], padding=[1], dout=0.25, name='pool1')

        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv21')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv22')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv23')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv24')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv25')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv26')
        n = dt.transform.pool(n, size=(2, 2), stride=[2, 2], padding=[1], dout=0.25, name='pool2')

        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv31')
        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv32')
        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv33')
        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv34')
        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv35')
        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv36')
        n = dt.transform.pool(n, size=(8, 8), stride=[8, 8], padding=[1], dout=0.25, name='pool3')

    # fc layers
    with dt.ctx(name='fcs', act='relu', bn=True, weight_filler='xavier'):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = dt.transform.flatten(n)
        n = dt.layer.dense(n, dim=500,
                regularizer=regularizer, weight_decay=fc_decay, dout=0.25, name='fc1')
        n = dt.layer.dense(n, dim=num_classes, act='linear', bn=False,
                regularizer=regularizer, weight_decay=fc_decay, name='fc2')

    return n

def fitnet1(in_tensor, num_classes):

    # conv layers
    with dt.ctx(name='convs', act='relu', bn=False, weight_filler='xavier'):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())

        regularizer=None
        conv_decay=0
        fc_decay=0

        n = in_tensor
        n = dt.layer.conv(n, size=(3, 3), dim=16, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv11')
        n = dt.layer.conv(n, size=(3, 3), dim=16, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv12')
        n = dt.layer.conv(n, size=(3, 3), dim=16, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv13')
        n = dt.transform.pool(n, size=(2, 2), stride=[2, 2], padding=[1], name='pool1')

        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv21')
        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv22')
        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv23')
        n = dt.transform.pool(n, size=(2, 2), stride=[2, 2], padding=[1], name='pool2')

        n = dt.layer.conv(n, size=(3, 3), dim=48, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv31')
        n = dt.layer.conv(n, size=(3, 3), dim=48, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv32')
        n = dt.layer.conv(n, size=(3, 3), dim=64, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv33')
        n = dt.transform.pool(n, size=(8, 8), stride=[8, 8], padding=[1], name='pool3')

    # fc layers
    with dt.ctx(name='fcs', act='relu', bn=False, weight_filler='xavier'):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = dt.transform.flatten(n)
        n = dt.layer.dense(n, dim=500,
                regularizer=regularizer, weight_decay=fc_decay, name='fc1')
        n = dt.layer.dense(n, dim=num_classes, act='linear', bn=False,
                regularizer=regularizer, weight_decay=fc_decay, name='fc2')

    return n

def fitnet2(in_tensor, num_classes):

    # conv layers
    with dt.ctx(name='convs', act='relu', bn=False, weight_filler='xavier'):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())

        regularizer=None
        conv_decay=0
        fc_decay=0

        n = in_tensor
        n = dt.layer.conv(n, size=(3, 3), dim=16, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv11')
        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv12')
        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv13')
        n = dt.transform.pool(n, size=(2, 2), stride=[2, 2], padding=[1], name='pool1')

        n = dt.layer.conv(n, size=(3, 3), dim=48, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv21')
        n = dt.layer.conv(n, size=(3, 3), dim=64, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv22')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv23')
        n = dt.transform.pool(n, size=(2, 2), stride=[2, 2], padding=[1], name='pool2')

        n = dt.layer.conv(n, size=(3, 3), dim=96, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv31')
        n = dt.layer.conv(n, size=(3, 3), dim=96, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv32')
        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv33')
        n = dt.transform.pool(n, size=(8, 8), stride=[8, 8], padding=[1], name='pool3')

    # fc layers
    with dt.ctx(name='fcs', act='relu', bn=False, weight_filler='xavier'):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = dt.transform.flatten(n)
        n = dt.layer.dense(n, dim=500,
                regularizer=regularizer, weight_decay=fc_decay, name='fc1')
        n = dt.layer.dense(n, dim=num_classes, act='linear', bn=False,
                regularizer=regularizer, weight_decay=fc_decay, name='fc2')

    return n

def fitnet3(in_tensor, num_classes):

    # conv layers
    with dt.ctx(name='convs', act='relu', bn=False, weight_filler='xavier'):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())

        regularizer=None
        conv_decay=0
        fc_decay=0

        n = in_tensor
        n = dt.layer.conv(n, size=(3, 3), dim=32, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv11')
        n = dt.layer.conv(n, size=(3, 3), dim=48, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv12')
        n = dt.layer.conv(n, size=(3, 3), dim=64, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv13')
        n = dt.layer.conv(n, size=(3, 3), dim=64, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv14')
        n = dt.transform.pool(n, size=(2, 2), stride=[2, 2], padding=[1], name='pool1')

        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv21')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv22')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv23')
        n = dt.layer.conv(n, size=(3, 3), dim=80, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv24')
        n = dt.transform.pool(n, size=(2, 2), stride=[2, 2], padding=[1], name='pool2')

        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv31')
        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv32')
        n = dt.layer.conv(n, size=(3, 3), dim=128, stride=[1, 1], padding=[1],
                          regularizer=regularizer, weight_decay=conv_decay, name='conv33')
        n = dt.transform.pool(n, size=(8, 8), stride=[8, 8], padding=[1], name='pool3')

    # fc layers
    with dt.ctx(name='fcs', act='relu', bn=False, weight_filler='xavier'):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = dt.transform.flatten(n)
        n = dt.layer.dense(n, dim=500,
                regularizer=regularizer, weight_decay=fc_decay, name='fc1')
        n = dt.layer.dense(n, dim=num_classes, act='linear', bn=False,
                regularizer=regularizer, weight_decay=fc_decay, name='fc2')

    return n


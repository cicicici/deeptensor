from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


def get_dim_stride(group, block, base_dim):
    dim = base_dim * (2 ** group)
    if group > 0 and block == 0:
        stride = 2
    else:
        stride = 1
    return dim, stride

def get_shortcut(group, block, x, out_dim, stride, identity, data_format, bn=True):
    in_dim = dt.tensor.get_dim(x, data_format)
    if in_dim == out_dim:
        if stride == 1:
            shortcut = x
        else: # should only be 2
            shortcut = dt.transform.pool(x, size=(stride, stride), stride=[stride, stride], name='sc_pool{}_{}'.format(group, block))
    else:
        if identity:
            if stride is not 1:
                x = dt.transform.pool(x, size=(stride, stride), stride=[stride, stride], name='sc_pool{}_{}'.format(group, block), avg=True)
            if in_dim < out_dim:
                pad_dim = (out_dim - in_dim) // 2
            paddings = dt.tensor.get_padding_channel([pad_dim], data_format)
            shortcut = tf.pad(x, paddings, name='sc_pad{}_{}'.format(group, block))
        else:
            shortcut = dt.layer.conv(x, size=(1, 1), dim=out_dim, stride=[stride, stride],
                                     bn=bn, ln=False, act=None, dout=0,
                                     name='sc_conv{}_{}'.format(group, block))
    return shortcut

def resnet_v1_basic_block(n, group, block, base_dim, identity, data_format, se_ratio):
    with dt.ctx(name='block{}_{}'.format(group, block)):
        dim, stride = get_dim_stride(group, block, base_dim)
        shortcut = get_shortcut(group, block, n, dim, stride, identity, data_format)

        n = dt.layer.conv(n, layer_first=True, shortcut=None,
                          size=(3, 3), dim=dim, stride=[stride, stride], name='res{}_{}_0'.format(group, block))
        n = dt.layer.conv(n, layer_first=True, shortcut=shortcut, bn_gamma=0,
                          size=(3, 3), dim=dim, stride=[1, 1], name='res{}_{}_1'.format(group, block))
        if se_ratio > 0:
            n_se = dt.model.se_block(n, dim, dim, data_format, se_ratio, name='se{}_{}'.format(group, block))
            n = n * n_se
    return n

def resnet_v1_bottleneck_block(n, group, block, base_dim, identity, data_format, se_ratio, stride_first=False, xt_width_ratio=1, xt_cardinality=0):
    with dt.ctx(name='block{}_{}'.format(group, block)):
        dim, stride = get_dim_stride(group, block, base_dim)
        in_dim = dim * xt_width_ratio if xt_cardinality > 0 else dim
        out_dim = dim * 4
        shortcut = get_shortcut(group, block, n, out_dim, stride, identity, data_format)

        n = dt.layer.conv(n, layer_first=True, shortcut=None,
                          size=(1, 1), dim=in_dim, stride=[stride, stride] if stride_first else [1, 1], name='res{}_{}_0'.format(group, block))
        n = dt.layer.conv(n, layer_first=True, shortcut=None, group=xt_cardinality,
                          size=(3, 3), dim=in_dim, stride=[1, 1] if stride_first else [stride, stride], name='res{}_{}_1'.format(group, block))
        n = dt.layer.conv(n, layer_first=True, shortcut=shortcut, bn_gamma=0,
                          size=(1, 1), dim=out_dim, stride=[1, 1], name='res{}_{}_2'.format(group, block))
        if se_ratio > 0:
            n_se = dt.model.se_block(n, in_dim, out_dim, data_format, se_ratio, name='se{}_{}'.format(group, block))
            n = n * n_se
    return n

def resnet_v1(in_tensor, num_classes,
              block_type='basic', blocks=[3, 3, 3],
              conv0_size=3, conv0_stride=1,
              pool0_size=0, pool0_stride=2,
              base_dim=16,
              regularizer='l2', conv_decay=1e-4, fc_decay=1e-4,
              shortcut='identity', weight_filler='variance', use_bias=False,
              data_format=dt.dformat.DEFAULT, se_ratio=0,
              xt_width_ratio=1, xt_cardinality=0):

    block_layers = 2
    if block_type == 'bottleneck':
        block_layers = 3

    dt.debug(dt.DC.NET, "[RESNET] v1, class {}, type {}, blocks {}, conv0 (d {}, s {}), pool0 (d {}, s {}), base (d {}), layers {}, se {}, width {}, card {}"
                                 .format(num_classes, block_type, blocks,
                                         conv0_size, conv0_stride, pool0_size, pool0_stride,
                                         base_dim, 1 + sum(blocks) * block_layers + 1,
                                         se_ratio, xt_width_ratio * base_dim, xt_cardinality))
    n = in_tensor
    # conv layers
    with dt.ctx(name='convs', act='relu', bn=True, weight_filler=weight_filler,
                regularizer=regularizer, weight_decay=conv_decay, bias=use_bias,
                pad='SAME', padding=None, data_format=data_format):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())

        n = dt.layer.conv(n, layer_first=True, shortcut=None,
                          size=(conv0_size, conv0_size), dim=base_dim, stride=[conv0_stride, conv0_stride], name='conv0')
        if pool0_size > 0:
            n = dt.transform.pool(n, size=(pool0_size, pool0_size), stride=[pool0_stride, pool0_stride], name='pool0')

        for group in range(len(blocks)):
            for block in range(blocks[group]):
                if block_type == 'bottleneck':
                    n = resnet_v1_bottleneck_block(n, group, block, base_dim, shortcut=='identity', data_format, se_ratio,
                                                   xt_width_ratio=xt_width_ratio, xt_cardinality=xt_cardinality)
                else:
                    n = resnet_v1_basic_block(n, group, block, base_dim, shortcut=='identity', data_format, se_ratio)

        n = dt.transform.pool(n, size=(8, 8), stride=[8, 8], name='pool1', avg=True)

    # fc layers
    with dt.ctx(name='fcs', act='relu', bn=True, weight_filler=weight_filler,
                regularizer=regularizer, weight_decay=fc_decay, bias=True, data_format=data_format):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = dt.transform.flatten(n, name='flatten')
        #n = dt.layer.dense(n, dim=256, name='fc1')
        n = dt.layer.dense(n, dim=num_classes, act='linear', bn=False, ln=False, name='fc2')

    return n

def resnet_v2_basic_block(n, group, block, base_dim, identity, data_format, se_ratio):
    with dt.ctx(name='block{}_{}'.format(group, block)):
        dim, stride = get_dim_stride(group, block, base_dim)
        shortcut = get_shortcut(group, block, n, dim, stride, identity, data_format)

        n = dt.layer.conv(n, layer_first=False, shortcut=None,
                          size=(3, 3), dim=dim, stride=[stride, stride], name='res{}_{}_0'.format(group, block))
        n = dt.layer.conv(n, layer_first=False, shortcut=shortcut,
                          size=(3, 3), dim=dim, stride=[1, 1], name='res{}_{}_1'.format(group, block))
        if se_ratio > 0:
            n_se = dt.model.se_block(n, dim, dim, data_format, se_ratio, name='se{}_{}'.format(group, block))
            n = n * n_se
    return n

def resnet_v2_bottleneck_block(n, group, block, base_dim, identity, data_format, se_ratio, stride_first=False, xt_width_ratio=1, xt_cardinality=0):
    with dt.ctx(name='block{}_{}'.format(group, block)):
        dim, stride = get_dim_stride(group, block, base_dim)
        in_dim = dim * xt_width_ratio if xt_cardinality > 0 else dim
        out_dim = dim * 4
        shortcut = get_shortcut(group, block, n, out_dim, stride, identity, data_format)

        n = dt.layer.conv(n, layer_first=False, shortcut=None,
                          size=(1, 1), dim=in_dim, stride=[stride, stride] if stride_first else [1, 1], name='res{}_{}_0'.format(group, block))
        n = dt.layer.conv(n, layer_first=False, shortcut=None, group=xt_cardinality,
                          size=(3, 3), dim=in_dim, stride=[1, 1] if stride_first else [stride, stride], name='res{}_{}_1'.format(group, block))
        n = dt.layer.conv(n, layer_first=False, shortcut=shortcut,
                          size=(1, 1), dim=out_dim, stride=[1, 1], name='res{}_{}_2'.format(group, block))
        if se_ratio > 0:
            n_se = dt.model.se_block(n, in_dim, out_dim, data_format, se_ratio, name='se{}_{}'.format(group, block))
            n = n * n_se
    return n

def resnet_v2(in_tensor, num_classes,
              block_type='basic', blocks=[3, 3, 3],
              conv0_size=3, conv0_stride=1,
              pool0_size=0, pool0_stride=2,
              base_dim=16,
              regularizer='l2', conv_decay=1e-4, fc_decay=1e-4,
              shortcut='identity', weight_filler='variance', use_bias=False,
              data_format=dt.dformat.DEFAULT, se_ratio=0,
              xt_width_ratio=1, xt_cardinality=0):

    block_layers = 2
    if block_type == 'bottleneck':
        block_layers = 3

    dt.debug(dt.DC.NET, "[RESNET] v2, class {}, type {}, blocks {}, conv0 (d {}, s {}), pool0 (d {}, s {}), base (d {}), layers {}, se {}, width {}, card {}"
                                 .format(num_classes, block_type, blocks,
                                         conv0_size, conv0_stride, pool0_size, pool0_stride,
                                         base_dim, 1 + sum(blocks) * block_layers + 1,
                                         se_ratio, xt_width_ratio * base_dim, xt_cardinality))
    n = in_tensor
    # conv layers
    with dt.ctx(name='convs', act='relu', bn=True, weight_filler=weight_filler,
                regularizer=regularizer, weight_decay=conv_decay, bias=use_bias,
                pad='SAME', padding=None, data_format=data_format):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())

        n = dt.layer.conv(n, layer_first=True, shortcut=None,
                          size=(conv0_size, conv0_size), dim=base_dim, stride=[conv0_stride, conv0_stride],
                          bn=False, ln=False, act=None, dout=0, name='conv0')
        if pool0_size > 0:
            n = dt.transform.pool(n, size=(pool0_size, pool0_size), stride=[pool0_stride, pool0_stride], name='pool0')

        for group in range(len(blocks)):
            for block in range(blocks[group]):
                if block_type == 'bottleneck':
                    n = resnet_v2_bottleneck_block(n, group, block, base_dim, shortcut=='identity', data_format, se_ratio,
                                                   xt_width_ratio=xt_width_ratio, xt_cardinality=xt_cardinality)
                else:
                    n = resnet_v2_basic_block(n, group, block, base_dim, shortcut=='identity', data_format, se_ratio)

        # apply bn, relu
        n = dt.layer.bypass(n, name='bnle0')
        n = dt.transform.pool(n, size=(8, 8), stride=[8, 8], name='pool1', avg=True)

    # fc layers
    with dt.ctx(name='fcs', act='relu', bn=True, weight_filler=weight_filler,
                regularizer=regularizer, weight_decay=fc_decay, bias=True, data_format=data_format):
        dt.log_pp(dt.DC.NET, dt.DL.DEBUG, dt.get_ctx())
        n = dt.transform.flatten(n, name='flatten')
        #n = dt.layer.dense(n, dim=256, name='fc1')
        n = dt.layer.dense(n, dim=num_classes, act='linear', bn=False, ln=False, name='fc2')

    return n


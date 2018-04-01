from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


@dt.dec_sugar_func
def accuracy(tensor, opt):

    assert opt.target is not None, 'target is mandatory.'

    if opt.name is None:
        acc_name = 'acc'
    else:
        acc_name = opt.name

    # # calc accuracy
    out = tf.identity(dt.transform.float(tf.equal(dt.transform.argmax(tensor), tf.cast(opt.target, tf.int64))), name=acc_name)

    return out

@dt.dec_sugar_func
def in_top_k(tensor, opt):

    assert opt.target is not None, 'target is mandatory.'
    opt += dt.Opt(k=1)

    if opt.name is None:
        top_k_name = 'in_top_{}'.format(opt.k)
    else:
        top_k_name = opt.name

    # # calc accuracy
    out = tf.identity(tf.cast(tf.nn.in_top_k(tensor, opt.target, opt.k), tf.float32), name=top_k_name)

    return out

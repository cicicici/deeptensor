from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


@dt.dec_sugar_func
def ce(tensor, opt):
    opt += dt.Opt(one_hot=False, softmax=False)
    assert opt.target is not None, 'target is mandatory.'

    dt.print_tensor_info(opt.target, "ce/labels")
    dt.print_tensor_info(tensor, "ce/logits")

    if opt.softmax:
        _epsilon = tf.convert_to_tensor(dt.eps, dt.floatx)
        tensor = tf.clip_by_value(tensor, _epsilon, 1 - _epsilon)
        tensor = tf.log(tensor)

    if opt.one_hot:
        ce = tf.identity(tf.nn.softmax_cross_entropy_with_logits(labels=opt.target, logits=tensor), 'ce')
    else:
        ce = tf.identity(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=opt.target, logits=tensor), 'ce')

    # masking loss
    if opt.mask:
        ce *= dt.transform.float(tf.not_equal(opt.target, tf.zeros_like(opt.target)))

    # add summary
    dt.summary_loss(ce, name=opt.name)

    ce_mean = tf.reduce_mean(ce, name='ce_mean')
    dt.summary_loss(ce_mean, name=opt.name)

    # return [batch] is ~3-5% faster than mean scalar
    return ce_mean


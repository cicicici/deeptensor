from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf


@dt.dec_sugar_func
def relu(x, opt):
    return tf.nn.relu(x, name=opt.name)

@dt.dec_sugar_func
def linear(x, opt):
    return x

@dt.dec_sugar_func
def softmax(x, opt):
    return tf.nn.softmax(x, name=opt.name)

act_list = {
    'relu': relu,
    'linear': linear,
    'softmax': softmax,
}

def perform(act, tensor, **kwargs):
    func = act_list[act]
    return func(tensor, **kwargs)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types

from functools import wraps
from contextlib import contextmanager
from tensorflow.python.client import device_lib

import deeptensor as dt
import tensorflow as tf


#
# GPU devices
#

_gpus = None

def gpus():
    global _gpus

    if _gpus is None:
        local_device_protos = device_lib.list_local_devices()
        _gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])

    return max(_gpus, 1)


_context = []

@contextmanager
def ctx(**kwargs):
    global _context

    # set options when enter
    # set options when enter
    context_now = dt.Opt(kwargs)
    _context += [context_now]

    # if named context
    if context_now.name:
        context_now.scope_name = context_now.name
        context_now.name = None
        with tf.variable_scope(context_now.scope_name):
            yield
    else:
        yield

    # clear options when exit
    del _context[-1]

def get_ctx():
    global _context

    # merge current context
    res = dt.Opt()
    for c in reversed(_context):
        res += c

    return res

def dec_sugar_func(func):

    @wraps(func)
    def wrapper(tensor, **kwargs):
        # kwargs parsing
        opt = dt.Opt(kwargs) + dt.get_ctx()

        # set default train mode
        opt += dt.Opt(is_training=True, reuse=None)

        # call sugar function
        out = func(tensor, opt)

        # save node info for reuse
        out._sugar = dt.Opt(func=func, arg=dt.Opt(kwargs)+get_ctx(),
                            prev=tensor)
        # inject reuse function
        out._reuse = types.MethodType(dt_reuse, out)

        return out

    return wrapper

def dt_reuse(tensor, **kwargs):
    opt = dt.Opt(kwargs)
    assert hasattr(tensor, '_sugar'), 'cannot reuse this node.'
    assert opt.input is not None, 'input is mandatory.'

    # get all nodes in this graph
    nodes, prev = [tensor], tensor._sugar.prev
    while prev is not None:
        nodes = [prev] + nodes
        prev = prev._sugar.prev if hasattr(prev, '_sugar') else None

    # create graph again for this input
    out = opt.input
    for node in nodes[1:]:  # exclude head node
        if node._sugar.is_layer:
            fn = dt.layer.dec_layer_func(node._sugar.func)
            if node._sugar.arg.scope_name:
                with tf.variable_scope(node._sugar.arg.scope_name):
                    out = fn(out, **(node._sugar.arg + dt.Opt(name=node._sugar.name, reuse=True)))
            else:
                out = fn(out, **(node._sugar.arg + dt.Opt(name=node._sugar.name, reuse=True)))
        else:
            fn = dt.dec_sugar_func(node._sugar.func)
            out = fn(out, **node._sugar.arg)

    return out

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps
from contextlib import contextmanager

import deeptensor as dt
import torch

#
# GPU devices
#

def gpus():
    return torch.cuda.device_count()

__global_ctx_list = []

@contextmanager
def ctx_cl(ctx_list, opt, **kwargs):
    global __global_ctx_list

    # append current context when enter
    if opt is None:
        _cur_ctx = dt.Opt(kwargs)
    else:
        _cur_ctx = opt + dt.Opt(kwargs)

    if ctx_list is None:
        __global_ctx_list += [_cur_ctx]
    else:
        ctx_list += [_cur_ctx]

    yield

    # clear current context when exit
    if ctx_list is None:
        del __global_ctx_list[-1]
    else:
        del ctx_list[-1]

@contextmanager
def ctx(**kwargs):
    global __global_ctx_list

    # append current context when enter
    _cur_ctx = dt.Opt(kwargs)
    __global_ctx_list += [_cur_ctx]

    yield

    # clear current context when exit
    del __global_ctx_list[-1]

def create_ctx_list(**kwargs):
    _cur_ctx = dt.Opt(kwargs)
    return [_cur_ctx]

def get_ctx_cl(ctx_list):
    global __global_ctx_list

    # merge current context
    res = dt.Opt()
    if ctx_list is None:
        for c in reversed(__global_ctx_list):
            res += c
    else:
        for c in reversed(ctx_list):
            res += c

    return res

def get_ctx():
    global __global_ctx_list

    # merge current context
    res = dt.Opt()
    for c in reversed(__global_ctx_list):
        res += c

    return res

def dec_ctx_func_cl(func):

    @wraps(func)
    def wrapper(ctx_list, **kwargs):
        # kwargs parsing
        _opt = dt.Opt(kwargs) + get_ctx_cl(ctx_list)

        _out = func(ctx_list, _opt)

        return _out

    return wrapper

def dec_ctx_func(func):

    @wraps(func)
    def wrapper(**kwargs):
        # kwargs parsing
        _opt = dt.Opt(kwargs) + get_ctx()

        _out = func(_opt)

        return _out

    return wrapper

def dec_sugar_func(func):

    @wraps(func)
    def wrapper(tensor, **kwargs):
        # kwargs parsing
        opt = dt.Opt(kwargs) + dt.get_ctx()

        # set default train mode
        opt += dt.Opt(is_training=True, reuse=None)
        # set default data format
        opt += dt.Opt(data_format=dt.dformat.DEFAULT)

        # call sugar function
        out = func(tensor, opt)

        return out

    return wrapper

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt


class Callback(object):

    def __init__(self, ctx, **kwargs):
        self._ctx = dt.Opt(kwargs) + ctx

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, value):
        self._idx = value

    def begin(self, **kwargs):
        pass

    def pre_step(self, **kwargs):
        return None

    def post_step(self, **kwargs):
        return None

    def end(self, **kwargs):
        pass


class CallGroup(object):

    def __init__(self, ctx, **kwargs):
        self._ctx = dt.Opt(kwargs) + ctx
        self._callbacks = []

    def add(self, callback):
        idx = len(self._callbacks)
        callback.idx = idx
        self._callbacks.append(callback)
        return idx

    def clear(self, clear):
        self._callbacks =[]

    def begin(self, **kwargs):
        for cb in self._callbacks:
            cb.begin(**kwargs)

    def pre_step(self, **kwargs):
        ret = []
        for cb in self._callbacks:
            ret.append(cb.pre_step(**kwargs))
        return ret

    def post_step(self, **kwargs):
        ret = []
        for cb in self._callbacks:
            ret.append(cb.post_step(**kwargs))
        return ret

    def end(self, **kwargs):
        for cb in self._callbacks:
            cb.end(**kwargs)

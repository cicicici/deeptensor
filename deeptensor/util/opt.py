from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import collections
import json

import deeptensor as dt


def opt_to_dict(opt):
    d = {}
    for k, v in six.iteritems(opt):
        if type(v) is Opt:
            d[k] = opt_to_dict(v)
        else:
            d[k] = v
    return d

def dict_to_opt(d):
    opt = dt.Opt()
    for k, v in d.items():
        if type(v) is dict:
            opt[k] = dict_to_opt(v)
        else:
            opt[k] = v
    return opt

class Opt(collections.MutableMapping):

    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __getattr__(self, key):
        return None

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    def __add__(self, other):
        res = Opt(self.__dict__)
        for k, v in six.iteritems(other):
            #if k not in res.__dict__ or res.__dict__[k] is None:
            if k not in res.__dict__:
                res.__dict__[k] = v
        return res

    def __mul__(self, other):
        res = Opt(self.__dict__)
        for k, v in six.iteritems(other):
            res.__dict__[k] = v
        return res

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def dumps(self):
        return json.dumps(opt_to_dict(self))

    def loads(self, s):
        res = Opt(self.__dict__)
        res *= dict_to_opt(json.loads(s))
        return res


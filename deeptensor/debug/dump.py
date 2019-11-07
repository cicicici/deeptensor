from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt

import pprint
pp = pprint.PrettyPrinter(indent=4)


def print_tensor_info(tensor, name="Tensor"):
    dt.debug(dt.DC.NET, "{}: shape {}".format(name, tensor.size()), frameskip=1)

def print_pp(obj, name="Obj"):
    global pp
    pp.pprint(obj)

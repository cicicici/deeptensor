from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod

import deeptensor as dt
import torch


class BaseData(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.tag = "DATA::BASE"
        dt.trace(dt.DC.DATA, "[{}] ({}) __init__".format(self.tag, type(self).__name__))

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = value

    # Abstract
    @abstractmethod
    def init_data(self):
        return self

    @abstractmethod
    def load_data(self):
        return self


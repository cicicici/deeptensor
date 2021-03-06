from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys

import logging
import inspect
import types
from typing import cast

import deeptensor as dt
from deeptensor.debug import DbgLvl as DL
from deeptensor.debug import DbgChn as DC
from deeptensor.debug import dbg_vld as DV

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


#this_function_name = cast(types.FrameType, inspect.currentframe()).f_code.co_name

class Logger(object):

    def __init__(self, log_file):
        self.log_file = log_file
        self.prefix = "[DT] "
        self.level = 10

    def setLevel(self, level):
        self.level = level

    def log(self, msg, *args, **kwargs):
        line = (self.prefix + msg).format(*args, **kwargs)
        print(line)

    def log_pp(self, *args, **kwargs):
        dt.print_pp(*args, **kwargs)


_logger_std = Logger(None)
_logger = logging.getLogger('deeptensor')
_logger.addHandler(logging.StreamHandler())
_logger.setLevel(logging.DEBUG)

_logger_tag = ''

def set_verbosity(verbosity):
    _logger.setLevel(verbosity)

def set_log_file(log_file):
    _logger.addHandler(logging.FileHandler(log_file))

def set_log_tag(tag):
    global _logger_tag
    _logger_tag = tag

def log(channel, level, msg, *args, **kwargs):
    if dt.DV(channel, level):
        _logger_std.log(msg, *args, **kwargs)

def log_pp(channel, level, msg, *args, **kwargs):
    if dt.DV(channel, level):
        _logger_std.log_pp(msg, *args, **kwargs)

def _log_prefix(marker="D", frameskip=0):

    # Returns (filename, line number) for the stack frame.
    def _get_file_line(frameskip):
        # pylint: disable=protected-access
        # noinspection PyProtectedMember
        f = sys._getframe()
        # pylint: enable=protected-access
        our_file = f.f_code.co_filename
        f = f.f_back
        while f:
            code = f.f_code
            if code.co_filename != our_file:
                if frameskip > 0:
                    frameskip -= 1
                else:
                    return code.co_filename, f.f_lineno
            f = f.f_back
        return '<unknown>', 0

    # current time
    now = time.time()
    now_tuple = time.localtime(now)
    now_microsecond = int(1e6 * (now % 1.0))

    # current filename and line
    filename, line = _get_file_line(frameskip)
    basename = os.path.basename(filename)

    s = '[%s%02d%02d_%02d:%02d:%02d.%06d %s/%s:%d] ' % (
        _logger_tag,
        now_tuple[1],  # month
        now_tuple[2],  # day
        now_tuple[3],  # hour
        now_tuple[4],  # min
        now_tuple[5],  # sec
        now_microsecond,
        marker,
        basename,
        line)

    return s

def trace(chn, msg, *args, frameskip=0, **kwargs):
    if dt.DV(chn, dt.DL.TRACE):
        _logger.debug(_log_prefix(marker="T", frameskip=frameskip) + msg, *args, **kwargs)

def debug(chn, msg, *args, frameskip=0, **kwargs):
    if dt.DV(chn, dt.DL.DEBUG):
        _logger.debug(_log_prefix(marker="D", frameskip=frameskip) + msg, *args, **kwargs)


def info(chn, msg, *args, frameskip=0, **kwargs):
    if dt.DV(chn, dt.DL.INFO):
        _logger.info(_log_prefix(marker="I", frameskip=frameskip) + msg, *args, **kwargs)


def warn(chn, msg, *args, frameskip=0, **kwargs):
    if dt.DV(chn, dt.DL.WARNING):
        _logger.warning(_log_prefix(marker="W", frameskip=frameskip) + msg, *args, **kwargs)


def error(chn, msg, *args, frameskip=0, **kwargs):
    if dt.DV(chn, dt.DL.ERROR):
        _logger.error(_log_prefix(marker="E", frameskip=frameskip) + msg, *args, **kwargs)


def critical(chn, msg, *args, frameskip=0, **kwargs):
    if dt.DV(chn, dt.DL.CRITICAL):
        _logger.critical(_log_prefix(marker="C", frameskip=frameskip) + msg, *args, **kwargs)

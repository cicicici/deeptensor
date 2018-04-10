#! /usr/bin/python
# -*- coding: utf8 -*-

import sys
import math

import deeptensor as dt
import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()

# Configuration
cfg = dt.util.Config(name="Scraper", app="scraper")
ARGS = cfg.opt().args

# Global
dt.info(dt.DC.STD, "data_dir: {}".format(ARGS.data_dir))



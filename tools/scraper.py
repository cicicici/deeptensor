#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import shutil
import re
import glob
import configparser
import json

import pandas as pd
from datetime import datetime

import deeptensor as dt
import horovod.tensorflow as hvd

hvd.init()

# Configuration
cfg = dt.util.Config(name="Scraper", app="scraper")
ARGS = cfg.opt().args

# Global
dt.info(dt.DC.STD, "data_dir: {}".format(ARGS.data_dir))

def clean_checkpoint(rank_path):
    with open(os.path.join(rank_path, "checkpoint")) as f:
        lines = f.readlines()
        last_model = None
        for line in lines:
            if line.startswith("model_checkpoint_path"):
                r = re.search('"(.+?)"', line)
                if r:
                    last_model = r.group(1)
                    #dt.debug(dt.DC.STD, "last_model: {}".format(last_model))

            if line.startswith("all_model_checkpoint_paths"):
                r = re.search('"(.+?)"', line)
                if r:
                    model = r.group(1)
                    if model != last_model:
                        for model_file in glob.glob(os.path.join(rank_path, model) + "*"):
                            dt.debug(dt.DC.STD, "Remove model fiel: {}".format(model_file))
                            os.remove(model_file)

def clean_model(data_dir, model_dir):
    for rank_dir in sorted(os.listdir(os.path.join(data_dir, model_dir))):
        if rank_dir.startswith("r") and rank_dir != "r0":
            dt.info(dt.DC.STD, "    Remove rank: {}".format(rank_dir))
            shutil.rmtree(os.path.join(data_dir, model_dir, rank_dir))
        elif rank_dir == "r0":
            clean_checkpoint(os.path.join(data_dir, model_dir, rank_dir))

def scan_line(stats, line):
    patterns = []
    patterns.append(dt.Opt(name='rank', filter=None,
                           fields=[dt.Opt(name='total', re='rank [0-9]+/(.+?),', type=int)]))
    patterns.append(dt.Opt(name='epoch', filter=' Epoch',
                           fields=[dt.Opt(name='ep', re='Epoch.([0-9]+?):', type=int),
                                   dt.Opt(name='lr', re='lr=([0-9.]+?):', type=float),
                                   dt.Opt(name='gs', re='gs=([0-9.]+?)]', type=float),
                                   dt.Opt(name='loss', re=' loss ([0-9.]+?),', type=float),
                                   dt.Opt(name='acc', re=' acc ([0-9.]+?),', type=float),
                                   dt.Opt(name='imgs', re=', ([0-9.]+?) img/s', type=float),
                                   dt.Opt(name='time', re='^\[([0-9-]+ [0-9:.]+?) ', type=str),
                                  ]))
    patterns.append(dt.Opt(name='valid', filter='valid\/acc1',
                           fields=[dt.Opt(name='ep', re='Epoch.([0-9]+?):', type=int),
                                   dt.Opt(name='lr', re='lr=([0-9.]+?):', type=float),
                                   dt.Opt(name='gs', re='gs=([0-9.]+?)]', type=float),
                                   dt.Opt(name='loss', re=' loss ([0-9.]+?),', type=float),
                                   dt.Opt(name='acc', re=' acc ([0-9.]+?),', type=float),
                                   dt.Opt(name='imgs', re=', ([0-9.]+?) img/s', type=float),
                                   dt.Opt(name='time', re='^\[([0-9-]+ [0-9:.]+?) ', type=datetime),
                                   dt.Opt(name='ce_mean', re='ce_mean ([0-9.]+?),', type=float),
                                   dt.Opt(name='acc1', re='acc1 ([0-9.]+?),', type=float),
                                   dt.Opt(name='top5', re='top5 ([0-9.]+?),', type=float),
                                  ]))

    for pat in patterns:
        if pat.filter is not None:
            r = re.search(pat.filter, line)
            if not r:
                continue

        if pat.name not in stats:
            stats[pat.name] = {}

        for fld in pat.fields:
            if fld.name not in stats[pat.name]:
                stats[pat.name][fld.name] = []

            r = re.search(fld.re, line)
            if r:
                match = r.group(1)
                if fld.type is int:
                    match = match.lstrip("0")
                elif fld.type is datetime:
                    match = datetime.strptime("2018-"+match, '%Y-%m-%d %H:%M:%S.%f')

                if fld.type is not str and fld.type is not datetime:
                    stats[pat.name][fld.name].append(json.loads(match))
                else:
                    stats[pat.name][fld.name].append(match)

def print_data(df):
    print(df)

def analyze_mode(data_dir, model_dir):
    rank_path = os.path.join(data_dir, model_dir, "r0")

    model_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    model_config.read(os.path.join(rank_path, "config.ini"))

    model_opt = dt.Opt()
    for section in model_config.sections():
        opt = dt.Opt()
        for key in model_config[section]:
            val_str =  model_config[section][key]
            val = json.loads(val_str)
            opt[key] = val
        model_opt[section] = opt

    model_stats = dt.Opt()
    with open(os.path.join(rank_path, "log.txt")) as f:
        lines = f.readlines()
        for line in lines:
            scan_line(model_stats, line)

    df_epoch = pd.DataFrame(data=model_stats.epoch)
    df_valid = pd.DataFrame(data=model_stats.valid)

    #print_data(df_epoch)
    #print_data(df_valid)

    fields = [model_dir.ljust(32),
              df_valid['acc1'].max(),
              df_valid['top5'].max(),
              df_valid['ep'].max(),
              df_valid['lr'].min(),
              df_epoch['acc'].max(),
              df_epoch['loss'].min(),
              df_epoch['imgs'].max(),
              model_stats.rank['total'][0],
              model_opt.args.lr_initial,
              model_opt.args.batch_size,
              model_opt.args.valid_size,
              model_opt.args.shuffle_size,
              model_opt.args.model_name,
              model_opt.args.model_type,
              model_opt.args.block_type,
              model_opt.args.shortcut,
              model_opt.args.blocks,
              model_opt.args.regularizer,
              model_opt.args.conv_decay,
              model_opt.args.fc_decay,
              model_opt.args.optim,
              model_opt.args.lr_curve,
              model_opt.args.class_num,
              model_opt.args.dataset,
              model_opt.args.data_type,
              model_opt.args.validate_ep,
              model_opt.args.deferred]

    fmt = ">"
    for fld in fields:
        fmt += "\t{}"
    print(fmt.format(*fields))

def scan_data(data_dir, data_filter):
    if not os.path.isdir(data_dir):
        dt.info(dt.DC.STD, "Invalid data dir: {}".format(data_dir))
        return

    for model_dir in sorted(os.listdir(data_dir)):
        r = re.search(data_filter, model_dir)
        if r:
            clean_model(data_dir, model_dir)
            analyze_mode(data_dir, model_dir)

scan_data(ARGS.data_dir, ARGS.data_filter)


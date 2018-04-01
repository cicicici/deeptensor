from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import argparse
import configparser
import json

import deeptensor as dt
import horovod.tensorflow as hvd


class Config(object):

    def __init__(self, name="demo", app='train', argv=None, command=None):
        self._name = name
        self._app = app
        if argv is None:
            self._argv = sys.argv[1:]
            self._command = sys.argv[0]
        else:
            self._argv = argv
            self._command = command

        self.init_args()
        self.parse_args()
        self.init_config()
        self.save_config()

    def init_args(self):
        self.args_parser = argparse.ArgumentParser(
            description="DeepTensor {}".format(self._name),
            epilog="Usage: {} -c config.ini [options]".format(self._command)
        )

        # required argument
        self.args_parser.add_argument('-c', type=str, action="store",
                                      help='Config file',
                                      default="config.ini", required=True)
        # optional arguments
        self.args_parser.add_argument('--name', type=str, help='Name')
        self.args_parser.add_argument('--tag', type=str, help='Tag')
        self.args_parser.add_argument('--host', type=str, help='Datalink host')
        self.args_parser.add_argument('--port', type=int, help='Datalink port')
        self.args_parser.add_argument('--out_dir', type=str, help='Output dir')
        self.args_parser.add_argument('--model_dir', type=str, help='Model dir')

        if self._app == 'train':
            self.args_parser.add_argument('--data_dir', type=str, help='Data dir')
            self.args_parser.add_argument('--data_type', type=str, help='Data type (tfrecord/folder)')
            self.args_parser.add_argument('--idx_file', type=str, help='Index file')
            self.args_parser.add_argument('--batch_size', type=int, help='Batch size')
            self.args_parser.add_argument('--valid_size', type=int, help='Valid size')
            self.args_parser.add_argument('--model_name', type=str, help='Model name (resnet)')
            self.args_parser.add_argument('--model_type', type=str, help='Model type (v1/v2)')
            self.args_parser.add_argument('--block_type', type=str, help='Block type (basic/bottleneck)')
            self.args_parser.add_argument('--blocks', type=str, help='Blocks ([3, 3, 3])')
            self.args_parser.add_argument('--regularizer', type=str, help='Regularizer type (l1/l2/"")')
            self.args_parser.add_argument('--conv_decay', type=float, help='Weight decay for convolution layers (1e-4)')
            self.args_parser.add_argument('--fc_decay', type=float, help='Weight decay for fully connected layers (1e-4)')
            self.args_parser.add_argument('--optim', type=str, help='Optimizer (Adam/MaxProp/...)')
            self.args_parser.add_argument('--lr_initial', type=float, help='Initial learning rate (0.001)')
            self.args_parser.add_argument('--lr_minimal', type=float, help='Minimal learning rate (1e-8)')
            self.args_parser.add_argument('--lr_curve', type=str, help='Learning reate curve ([[0.1, 80, 1]])')
            self.args_parser.add_argument('--shortcut', type=str, help='Shortcut type (identity/1x1conv)')
            self.args_parser.add_argument('--class_num', type=int, help='Number of classes (10/100/1000)')
            self.args_parser.add_argument('--class_min', type=int, help='Minimal index of the first class (0)')
            self.args_parser.add_argument('--validate', type=str, help='Enable online validation (true/false)')

    def parse_args(self):
        self._args = self.args_parser.parse_args(self._argv)
        #print(self._args)

    def init_config(self):
        self._config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self._config.read(self._args.c)

        self._opt = dt.Opt()

        for section in self._config.sections():
            opt = dt.Opt()
            for key in self._config[section]:
                val_str =  self._config[section][key]
                val = json.loads(val_str)
                opt[key] = val
            self._opt[section] = opt

        for arg in vars(self._args):
            val = getattr(self._args, arg)
            if val is None:
                continue
            val_str = str(val)

            if arg in self._opt.args:
                opt_val = self._opt.args[arg]
                if isinstance(opt_val, str):
                    val_str = '"' + val_str + '"'

                if type(opt_val) is not type(val):
                    print("[Convert Arg] {}: {}, {} => {}".format(arg, val, type(val), type(opt_val)))
                    self._opt.args[arg] = json.loads(val_str)
                else:
                    self._opt.args[arg] = val
                self._config['args'][arg] = val_str

    def save_config(self):
        model_dir = self._opt.args.model_dir
        print(model_dir)
        if not (model_dir is not None and model_dir.startswith('/')):
            if model_dir is None or len(model_dir) == 0:
                model_dir=time.strftime('%Y%m%d_%H%M%S', time.localtime())
            if len(self._opt.args.tag) > 0:
                model_dir = model_dir + "_" + self._opt.args.tag
            model_dir="{}/{}".format(self._opt.args.out_dir, model_dir)

        model_dir = "{}/r{}".format(model_dir, hvd.rank())

        try:
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
        except:
            pass
        self._opt.args.model_dir = model_dir
        self._config['args']['model_dir'] = '"' + model_dir + '"'
        # config file
        with open('{}/config.ini'.format(model_dir), 'w') as configfile:
            self._config.write(configfile)
        # log file
        dt.set_log_file('{}/log.txt'.format(model_dir))

    def dump_config(self):
        dt.log(dt.DC.STD, dt.DL.INFO, "{} - {}".format(self._opt.args.name, self._opt.args.tag))
        dt.print_pp(self._opt)
        if self._app == 'train':
            dt.log(dt.DC.NET, dt.DL.INFO, "Tensorboard: $ tensorboard --logdir {} --port {}".format(self._opt.args.model_dir, 6006))

    def opt(self):
        return self._opt;


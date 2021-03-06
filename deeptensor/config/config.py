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
import horovod.torch as hvd

class Config(object):

    def __init__(self, name="deeptensor", app='train', argv=None, command=None):
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
        self.default_config()
        self.init_config()
        self.save_config()
        self.post_config()

    def init_usr_args(self):
        pass

    def init_args(self):
        self.args_parser = argparse.ArgumentParser(
            description="DeepTensor {}".format(self._name),
            epilog="Usage: {} -c config.ini [options]".format(self._command)
        )

        self.init_usr_args()

        # optional arguments
        self.args_parser.add_argument('-c', type=str, action="store",
                                      help='Config file',
                                      default="config.ini", required=True)
        # optional arguments
        self.args_parser.add_argument('--name', type=str, help='Name')
        self.args_parser.add_argument('--tag', type=str, help='Tag')
        self.args_parser.add_argument('--host', type=str, help='Datalink host')
        self.args_parser.add_argument('--port', type=int, help='Datalink port')
        self.args_parser.add_argument('--work_dir', type=str, help='Output dir')
        self.args_parser.add_argument('--model_dir', type=str, help='Model dir')
        self.args_parser.add_argument('--add', type=str, help='Addtitional options')
        self.args_parser.add_argument('-m', type=str, help='Run mode')
        self.args_parser.add_argument('--trace', action='store_const', const=True, help='Enable tracing')

        if self._app == 'train':
            self.args_parser.add_argument('--data_dir', type=str, help='Data dir')
            self.args_parser.add_argument('--data_type', type=str, help='Data type (tfrecord/folder)')
            self.args_parser.add_argument('--idx_file', type=str, help='Index file')
            self.args_parser.add_argument('--batch_size', type=int, help='Batch size')
            self.args_parser.add_argument('--valid_size', type=int, help='Valid size')
            self.args_parser.add_argument('--out_size', type=int, help='Output size')
            self.args_parser.add_argument('--num_workers', type=int, help='Worker threads for dataset')
            self.args_parser.add_argument('--data_format', type=str, help='Data format (NHWC/NCHW)')
            self.args_parser.add_argument('--model_name', type=str, help='Model name (resnet)')
            self.args_parser.add_argument('--model_type', type=str, help='Model type (v1/v2)')
            self.args_parser.add_argument('--block_type', type=str, help='Block type (basic/bottleneck)')
            self.args_parser.add_argument('--blocks', type=str, help='Blocks ([3, 3, 3])')
            self.args_parser.add_argument('--regularizer', type=str, help='Regularizer type (l1/l2/"")')
            self.args_parser.add_argument('--pretrained', type=int, help='Pretrained model')
            self.args_parser.add_argument('--optim', type=str, help='Optimizer (Adam/MaxProp/...)')
            self.args_parser.add_argument('--lr_initial', type=float, help='Initial learning rate (0.001)')
            self.args_parser.add_argument('--lr_minimal', type=float, help='Minimal learning rate (1e-8)')
            self.args_parser.add_argument('--lr_curve', type=str, help='Learning reate curve ([[0.1, 80, 1]])')
            self.args_parser.add_argument('--momentum', type=float, help='Optimizer momentum (0.9)')
            self.args_parser.add_argument('--alpha', type=float, help='Optimizer alpha (0.9)')
            self.args_parser.add_argument('--beta1', type=float, help='Optimizer beta1 (0.9)')
            self.args_parser.add_argument('--beta2', type=float, help='Optimizer beta2 (0.99)')
            self.args_parser.add_argument('--opt_eps', type=float, help='Optimizer eps (1e-6)')
            self.args_parser.add_argument('--class_num', type=int, help='Number of classes (10/100/1000)')
            self.args_parser.add_argument('--class_min', type=int, help='Minimal index of the first class (0)')
            self.args_parser.add_argument('--validate_ep', type=int, help='Validate every [n] epochs, set 0 to disable')
            self.args_parser.add_argument('--max_ep', type=int, help='Maximum epochs to run, (default: 1000)')
            self.args_parser.add_argument('--gpu0', type=int, help='GPU index for device 0, (default: 0)')
            self.args_parser.add_argument('--valid_only', action='store_const', const=True, help='Validation only')
        elif self._app == 'scraper':
            self.args_parser.add_argument('--data_dir', type=str, help='Data dir')
            self.args_parser.add_argument('--data_filter', type=str, help='Data dir filter')

    def parse_args(self):
        self._args = self.args_parser.parse_args(self._argv)

    def default_set_config(self, section, key, val):
        self._default_config[section][key] = val

    def default_usr_config(self):
        pass

    def default_config(self):
        self._default_config = {}

        section = 'args'
        self._default_config[section] = {}
        # common configurations
        self._default_config[section]['name'] = "Default"
        self._default_config[section]['tag'] = ""
        self._default_config[section]['host'] = "127.0.0.1"
        self._default_config[section]['port'] = 7001
        self._default_config[section]['work_dir'] = "_train/default"
        self._default_config[section]['model_dir'] = ""
        self._default_config[section]['add'] = {}
        self._default_config[section]['m'] = ""
        self._default_config[section]['trace'] = False

        # training configurations
        if self._app == 'train':
            self._default_config[section]['data_dir'] = "/datasets/imagenet"
            self._default_config[section]['data_type'] = "folder"
            self._default_config[section]['idx_file'] = ""
            self._default_config[section]['batch_size'] = 32
            self._default_config[section]['valid_size'] = 32
            self._default_config[section]['out_size'] = 224
            self._default_config[section]['num_workers'] = 1
            self._default_config[section]['data_format'] = "NCHW"
            self._default_config[section]['model_name'] = ""
            self._default_config[section]['model_type'] = ""
            self._default_config[section]['block_type'] = ""
            self._default_config[section]['blocks'] = []
            self._default_config[section]['regularizer'] = ""
            self._default_config[section]['pretrained'] = 0
            self._default_config[section]['optim'] = "Momentum"
            self._default_config[section]['lr_initial'] = 0.1
            self._default_config[section]['lr_minimal'] = 1e-8
            self._default_config[section]['lr_curve'] = [[0.1, 30, 3], [0.1, 20, 1]]
            self._default_config[section]['momentum'] = 0.9
            self._default_config[section]['alpha'] = 0.9
            self._default_config[section]['beta1'] = 0.9
            self._default_config[section]['beta2'] = 0.99
            self._default_config[section]['opt_eps'] = 1e-6
            self._default_config[section]['class_num'] = 1000
            self._default_config[section]['class_min'] = 0
            self._default_config[section]['validate_ep'] = 10
            self._default_config[section]['max_ep'] = 1000
            self._default_config[section]['gpu0'] = 0
            self._default_config[section]['valid_only'] = False

            # no command line argument
            self._default_config[section]['save_interval'] = 1
            self._default_config[section]['dataset'] = "default"
            self._default_config[section]['shuffle_size'] = 2048
        elif self._app == 'scraper':
            self._default_config[section]['data_dir'] = "_train/default"
            self._default_config[section]['data_filter'] = ".+"

        # debug configurations
        section = 'debug'
        self._default_config[section] = {}
        self._default_config[section]['channel'] = dt.DC.ALL
        self._default_config[section]['level'] = dt.DL.DEBUG

        self.default_usr_config()

    def init_config(self):
        self._config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self._config.read(self._args.c)

        if 'args' not in self._config:
            self._config['args'] = {}
        if 'debug' not in self._config:
            self._config['debug'] = {}

        self._opt = dt.Opt()

        # load config to opt
        for section in self._config.sections():
            opt = dt.Opt()
            for key in self._config[section]:
                val_str =  self._config[section][key]
                val = json.loads(val_str)
                opt[key] = val
            self._opt[section] = opt

        # override with command line args
        for arg in vars(self._args):
            val = getattr(self._args, arg)
            if val is None:
                continue
            val_str = str(val)

            if arg in self._opt.args or arg in self._default_config['args']:
                if arg in self._opt.args:
                    opt_val = self._opt.args[arg]
                else:
                    opt_val = self._default_config['args'][arg]

                if isinstance(opt_val, str):
                    val_str = '"' + val_str + '"'

                if type(opt_val) is not type(val):
                    dt.log(log.DC.STD, "[Convert Arg] {}: {}, {} => {}".format(arg, val, type(val), type(opt_val)))
                    self._opt.args[arg] = json.loads(val_str)
                else:
                    self._opt.args[arg] = val
                self._config['args'][arg] = val_str
            else:
                self._opt.args[arg] = val
                self._config['args'][arg] = val_str

        # add default settings
        for section in self._default_config:
            opt = dt.Opt()
            for key in self._default_config[section]:
                if key not in self._opt[section]:
                    opt[key] = self._default_config[section][key]
                    self._config[section][key] = json.dumps(opt[key])
            self._opt[section] += opt

        # additional post process
        if self._opt.args.add is not None and type(self._opt.args.add) is dict:
            self._opt.args.add = dt.util.dict_to_opt(self._opt.args.add)

    def save_config(self):
        model_dir = self._opt.args.model_dir
        if not (model_dir is not None and model_dir.startswith('/')):
            if model_dir is None or len(model_dir) == 0:
                model_dir=time.strftime('%Y%m%d_%H%M%S', time.localtime())
            if len(self._opt.args.tag) > 0:
                model_dir = model_dir + "_" + self._opt.args.tag
            model_dir="{}/{}".format(self._opt.args.work_dir, model_dir)

        inst_dir = "{}/r{}".format(model_dir, hvd.rank())

        try:
            if not os.path.isdir(inst_dir):
                os.makedirs(inst_dir)
        except:
            pass
        self._opt.args.model_dir = model_dir
        self._config['args']['_model_dir'] = '"' + model_dir + '"'
        self._opt.args.inst_dir = inst_dir
        self._config['args']['_inst_dir'] = '"' + inst_dir + '"'
        # config file
        with open('{}/config.ini'.format(inst_dir), 'w') as configfile:
            self._config.write(configfile)
        # log file
        dt.set_log_file('{}/log.txt'.format(inst_dir))
        dt.set_log_tag("{} ".format(hvd.rank()))

        self._model_dir = model_dir
        self._inst_dir = inst_dir

    def post_config(self):
        # Set debug settings
        dt.dbg_cfg(level=self._opt.debug.level,
                   channel=self._opt.debug.channel)

        if self._opt.args.trace and not dt.dbg_lvl(dt.DL.TRACE):
            dt.dbg_cfg(level=dt.DL.TRACE)

        # dump important information
        dt.info(dt.DC.STD, "{} - {}".format(self._opt.args.name, self._opt.args.tag))
        dt.info(dt.DC.STD, "Rank {}: command [{}], argv [{}]".format(hvd.rank(), self._command, self._argv))
        dt.info(dt.DC.STD, "Rank {}: model_dir [{}]".format(hvd.rank(), self._model_dir))
        dt.info(dt.DC.STD, "Rank {}: opt [{}]".format(hvd.rank(), self._opt))

    def opt(self):
        return self._opt;


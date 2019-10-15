#! /usr/bin/python
# -*- coding: utf8 -*-

import deeptensor as dt
import torch
import horovod.torch as hvd

# Init horovod
hvd.init()

# Configuration
cfg = dt.config.Config(name="MNIST")
ARGS = cfg.opt().args

class MnistEstimator(dt.estimator.ClassEstimator):
    def __init__(self, opt, cfg):
        super(MnistEstimator, self).__init__(opt, cfg)
        dt.debug(dt.DC.TRAIN, "[EST] {} initialized".format(type(self).__name__))

    def build_data(self, is_training):
        args = self._opt.args
        data = dt.data.Mnist(train_batch_size=args.batch_size, valid_batch_size=args.valid_size).init_data()
        ep_size = data.train.num_batch
        v_ep_size = data.valid.num_batch
        return dt.Opt(data=data, ep_size=ep_size, v_ep_size=v_ep_size)

    def build_model(self):
        # Params
        # args.model_name == "resnet":
        # args.model_type == "v1":
        # args.class_num
        # args.block_type
        # args.blocks
        # args.shortcut
        # args.regularizer
        # args.conv_decay
        # args.fc_decay
        # self._opt.data_format
        # args.se_ratio)

        return None

# Train
with dt.ctx(optim=ARGS.optim, lr_initial=ARGS.lr_initial, lr_minimal=ARGS.lr_minimal,
            lr_curve=ARGS.lr_curve):
    dt.train.train(args=ARGS, est_class = MnistEstimator, est_cfg=dt.Opt(),
                   batch_size=ARGS.batch_size, summary_freq=ARGS.summary_freq,
                   validate_ep=ARGS.validate_ep, max_ep=ARGS.max_ep,
                   model_dir=ARGS.model_dir, save_interval=ARGS.save_interval,
                   beta1=ARGS.beta1, beta2=ARGS.beta2, momentum=ARGS.momentum,
                   random_seed=1234 * (hvd.rank()+1), deferred=ARGS.deferred)


#! /usr/bin/python
# -*- coding: utf8 -*-

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import horovod.torch as hvd

# Init horovod
hvd.init()

# Configuration
cfg = dt.config.Config(name="MNIST")
ARGS = cfg.opt().args

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MnistEstimator(dt.estimator.ClassEstimator):
    def __init__(self, opt, cfg):
        super(MnistEstimator, self).__init__(opt, cfg)
        self.tag = "EST::MNIST"
        dt.trace(dt.DC.MODEL, "[{}] ({}) __init__".format(self.tag, type(self).__name__))

    def build_data(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build data".format(self.tag, type(self).__name__))
        args = self._ctx.args
        self._data = dt.data.Mnist(batch_size=args.batch_size, valid_size=args.valid_size,
                                   num_workers=1, pin_memory=self.use_cuda).init_data()
        return True

    def build_model(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build model".format(self.tag, type(self).__name__))

        self._model = Net().to(self.device)

        return True


# Train
with dt.ctx(optim=ARGS.optim, lr_initial=ARGS.lr_initial, lr_minimal=ARGS.lr_minimal, lr_curve=ARGS.lr_curve):
    dt.train.train(args=ARGS, est_class = MnistEstimator, est_cfg=dt.Opt(),
                   batch_size=ARGS.batch_size, valid_size=ARGS.valid_size, summary_freq=ARGS.summary_freq,
                   validate_ep=ARGS.validate_ep, max_ep=ARGS.max_ep,
                   model_dir=ARGS.model_dir, save_interval=ARGS.save_interval,
                   beta1=ARGS.beta1, beta2=ARGS.beta2, momentum=ARGS.momentum,
                   random_seed=1 * (hvd.rank()+1), deferred=ARGS.deferred)


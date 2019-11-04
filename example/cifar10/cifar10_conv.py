#! /usr/bin/python
# -*- coding: utf8 -*-

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import horovod.torch as hvd

# Init horovod
dt.train.init_library()

# Configuration
cfg = dt.config.Config(name="CIFAR-10")
ARGS = cfg.opt().args

# Datalink over network
#def datalink_recv(socket, packet):
#    opt = dt.Opt().loads(packet._data.decode())
#    #print(opt)
#    # Set learning rate
#    if opt.t == 'cmd':
#        if opt.a == 'set':
#            if opt.key == 'lr':
#                dt.train.set_lr_val(opt.val)

#    dt.util.datalink().send_opt_sock(socket, dt.Opt(ACK='OK'))

#if ARGS.port > 1000 and hvd.rank() == 0:
#    dt.util.datalink_start(port=ARGS.port)
#    dt.util.datalink_register_recv(datalink_recv)

class Cifar10Estimator(dt.estimator.ClassEstimator):
    def __init__(self, opt, cfg):
        super(Cifar10Estimator, self).__init__(opt, cfg)
        self.tag = "EST::CIFAR10"
        dt.trace(dt.DC.MODEL, "[{}] ({}) __init__".format(self.tag, type(self).__name__))

    def build_data(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build data".format(self.tag, type(self).__name__))
        args = self._ctx.args
        data = dt.data.Cifar10(batch_size=args.batch_size, valid_size=args.valid_size,
                               num_workers=1, pin_memory=self.use_cuda)
        data.init_data()
        data.load_data()
        self._data = data
        return True

    def build_model(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build model".format(self.tag, type(self).__name__))

        #self._model = dt.model.VGG('VGG19')
        #self._model = dt.model.ResNet18()
        #self._model = dt.model.ResNet152()
        #self._model = dt.model.PreActResNet18()
        #self._model = dt.model.GoogLeNet()
        #self._model = dt.model.DenseNet121()
        #self._model = dt.model.ResNeXt29_2x64d()
        #self._model = dt.model.MobileNet()
        #self._model = dt.model.MobileNetV2()
        #self._model = dt.model.DPN92()
        #self._model = dt.model.ShuffleNetG2()
        #self._model = dt.model.SENet18()
        #self._model = dt.model.ShuffleNetV2(1)
        self._model = dt.model.EfficientNetB0()

        return True

# Train
with dt.ctx(optim=ARGS.optim, data_format=ARGS.data_format,
            lr_initial=ARGS.lr_initial, lr_minimal=ARGS.lr_minimal, lr_curve=ARGS.lr_curve):
    dt.train.train(args=ARGS, est_class = Cifar10Estimator, est_cfg=dt.Opt(),
                   batch_size=ARGS.batch_size, valid_size=ARGS.valid_size,
                   validate_ep=ARGS.validate_ep, max_ep=ARGS.max_ep,
                   model_dir=ARGS.model_dir, save_interval=ARGS.save_interval,
                   beta1=ARGS.beta1, beta2=ARGS.beta2, momentum=ARGS.momentum, weight_decay=ARGS.weight_decay,
                   random_seed=1 * (hvd.rank()+1), gpu0=ARGS.gpu0)

#dt.util.datalink_close()


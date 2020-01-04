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
    def __init__(self, ctx):
        super(Cifar10Estimator, self).__init__(ctx)
        self.tag = "EST::CIFAR10"
        dt.trace(dt.DC.MODEL, "[{}] ({}) __init__".format(self.tag, type(self).__name__))

    def build_data(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build data".format(self.tag, type(self).__name__))
        args = self._ctx.args
        data = dt.data.Cifar10(data_dir='/datasets/cifar10',
                               batch_size=args.batch_size, valid_size=args.valid_size,
                               num_workers=1, pin_memory=self.use_cuda)
        data.init_data()
        data.load_data()
        self._data = data
        return True

    def build_model(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build model".format(self.tag, type(self).__name__))

        #self._model = dt.model.cifar.VGG('VGG19')        # target 92.64%
        self._model = dt.model.cifar.ResNet18()          # target 93.02%
        #self._model = dt.model.cifar.ResNet50()          # target 93.62%
        #self._model = dt.model.cifar.ResNet101()         # target 93.75%
        #self._model = dt.model.cifar.ResNet152()         # 8-gpu  94.2+%
        #self._model = dt.model.cifar.PreActResNet18()    # target 95.11%, NAN
        #self._model = dt.model.cifar.GoogLeNet()
        #self._model = dt.model.cifar.DenseNet121()       # target 95.04%
        #self._model = dt.model.cifar.ResNeXt29_32x4d()   # target 94.73%
        #self._model = dt.model.cifar.ResNeXt29_2x64d()   # target 94.82%
        #self._model = dt.model.cifar.MobileNet()
        #self._model = dt.model.cifar.MobileNetV2()       # target 94.43%
        #self._model = dt.model.cifar.DPN92()             # target 95.16%
        #self._model = dt.model.cifar.ShuffleNetG2()
        #self._model = dt.model.cifar.SENet18()
        #self._model = dt.model.cifar.ShuffleNetV2(1)
        #self._model = dt.model.cifar.EfficientNetB0()

        return True

# Train
ctx = dt.Opt(args=ARGS,
             optim=ARGS.optim, data_format=ARGS.data_format,
             lr_initial=ARGS.lr_initial, lr_minimal=ARGS.lr_minimal, lr_curve=ARGS.lr_curve,
             batch_size=ARGS.batch_size, valid_size=ARGS.valid_size,
             validate_ep=ARGS.validate_ep, max_ep=ARGS.max_ep,
             model_dir=ARGS.model_dir, save_interval=ARGS.save_interval,
             beta1=ARGS.beta1, beta2=ARGS.beta2, momentum=ARGS.momentum, weight_decay=ARGS.weight_decay,
             random_seed=1 * (hvd.rank()+1), gpu0=ARGS.gpu0, valid_only=ARGS.valid_only)

est = Cifar10Estimator(ctx)
est.build_flow()

trainer = dt.train.Trainer(ctx)
trainer.init()

trainer.bind_estimator(est)

trainer.train_setup()
trainer.train_begin()
trainer.train(max_ep = 2)
trainer.train_end()

#dt.util.datalink_close()

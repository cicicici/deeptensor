#! /usr/bin/python
# -*- coding: utf8 -*-

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import horovod.torch as hvd

import warnings

warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)

# Init horovod
dt.train.init_library()

# Configuration
cfg = dt.config.Config(name="ImageNet")
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

class ImageNetEstimator(dt.estimator.ClassEstimator):
    def __init__(self, opt, cfg):
        super(ImageNetEstimator, self).__init__(opt, cfg)
        self.tag = "EST::IMAGENET"
        dt.trace(dt.DC.MODEL, "[{}] ({}) __init__".format(self.tag, type(self).__name__))

    def build_data(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build data".format(self.tag, type(self).__name__))
        args = self._ctx.args
        data = dt.data.ImageNet(data_dir='/datasets/imagenet',
                                batch_size=args.batch_size, valid_size=args.valid_size,
                                out_size=224, num_workers=4, pin_memory=self.use_cuda)
        data.init_data()
        data.load_data()
        self._data = data
        return True

    def build_model(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build model".format(self.tag, type(self).__name__))

        #if dt.train.is_chief():
        #    dt.print_pp(torchvision.models.__dict__)

        #arch = 'alexnet'
        #arch = 'densenet121'
        #arch = 'densenet161'
        #arch = 'densenet169'
        #arch = 'densenet201'
        #arch = 'googlenet'
        #arch = 'inception_v3'
        #arch = 'mnasnet0_5'
        #arch = 'mnasnet0_75'
        #arch = 'mnasnet1_0'
        #arch = 'mnasnet1_3'
        #arch = 'mobilenet_v2'
        #arch = 'resnet18'
        #arch = 'resnet34'
        # arch = 'resnet50'
        #arch = 'resnet101'
        #arch = 'resnet152'
        #arch = 'resnext50_32x4d'
        #arch = 'resnext101_32x8d'
        #arch = 'shufflenet_v2_x0_5'
        #arch = 'shufflenet_v2_x1_0'
        #arch = 'shufflenet_v2_x1_5'
        #arch = 'shufflenet_v2_x2_0'
        #arch = 'squeezenet1_0'
        #arch = 'squeezenet1_1'
        #arch = 'vgg11'
        #arch = 'vgg11_bn'
        #arch = 'vgg13'
        #arch = 'vgg13_bn'
        #arch = 'vgg16'
        #arch = 'vgg16_bn'
        #arch = 'vgg19'
        #arch = 'vgg19_bn'
        #arch = 'wide_resnet50_2'
        #arch = 'wide_resnet101_2'

        #pretrained = True
        pretrained = False

        #self._model = torchvision.models.__dict__[arch](pretrained=pretrained)
        #dt.info(dt.DC.TRAIN, "arch {}, pretrained {}".format(arch, pretrained))

        #self._model = dt.model.imagenet.FairNasA()         # 8-gpu

        arch = 'efficientnet-b0'
        if pretrained:
            self._model = dt.model.EfficientNet.from_pretrained(arch)
        else:
            self._model = dt.model.EfficientNet.from_name(arch)
        dt.info(dt.DC.TRAIN, "arch {}, pretrained {}".format(arch, pretrained))

        return True

    def post_model(self):
        if dt.train.is_chief():
            dt.summary.summary_model_patch(self._model)
            dt.info(dt.DC.TRAIN, "\n{}".format(dt.summary.summary_model_fwd(self._model, (3, 224, 224), device='cpu')))
            dt.summary.summary_model_patch(self._model, patch_fn=dt.summary.patch_clear_dt)

# Train
with dt.ctx(optim=ARGS.optim, data_format=ARGS.data_format,
            lr_initial=ARGS.lr_initial, lr_minimal=ARGS.lr_minimal, lr_curve=ARGS.lr_curve):
    dt.train.train(args=ARGS, est_class=ImageNetEstimator, est_cfg=dt.Opt(),
                   batch_size=ARGS.batch_size, valid_size=ARGS.valid_size,
                   validate_ep=ARGS.validate_ep, max_ep=ARGS.max_ep,
                   model_dir=ARGS.model_dir, save_interval=ARGS.save_interval,
                   beta1=ARGS.beta1, beta2=ARGS.beta2, momentum=ARGS.momentum, weight_decay=ARGS.weight_decay,
                   random_seed=1 * (hvd.rank()+1), gpu0=ARGS.gpu0, valid_only=ARGS.valid_only)

#dt.util.datalink_close()

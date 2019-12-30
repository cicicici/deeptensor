#! /usr/bin/python
# -*- coding: utf8 -*-

import warnings

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import horovod.torch as hvd


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
        data = dt.data.ImageNet(data_dir=args.data_dir,
                                batch_size=args.batch_size, valid_size=args.valid_size,
                                out_size=args.out_size, num_workers=args.num_workers, pin_memory=self.use_cuda)
        data.init_data()
        data.load_data()
        self._data = data
        return True

    def build_model(self):
        dt.trace(dt.DC.MODEL, "[{}] ({}) build model".format(self.tag, type(self).__name__))
        args = self._ctx.args
        pretrained = (args.pretrained > 0)

        if args.model_name == 'efficientnet':
            if args.model_type == 'b0':
                self._model = dt.model.efficientnet.efficientnet_b0(pretrained=pretrained)
            elif args.model_type == 'b1':
                self._model = dt.model.efficientnet.efficientnet_b1(pretrained=pretrained)
            elif args.model_type == 'b2':
                self._model = dt.model.efficientnet.efficientnet_b2(pretrained=pretrained)
            elif args.model_type == 'b3':
                self._model = dt.model.efficientnet.efficientnet_b3(pretrained=pretrained)
            elif args.model_type == 'b4':
                self._model = dt.model.efficientnet.efficientnet_b4(pretrained=pretrained)
            elif args.model_type == 'b5':
                self._model = dt.model.efficientnet.efficientnet_b5(pretrained=pretrained)
            elif args.model_type == 'b6':
                self._model = dt.model.efficientnet.efficientnet_b6(pretrained=pretrained)
            elif args.model_type == 'b7':
                self._model = dt.model.efficientnet.efficientnet_b7(pretrained=pretrained)
        elif args.model_name == 'efficientnet_lm':
            if args.model_type == 'b0' or \
               args.model_type == 'b1' or \
               args.model_type == 'b2' or \
               args.model_type == 'b3' or \
               args.model_type == 'b4' or \
               args.model_type == 'b5' or \
               args.model_type == 'b6' or \
               args.model_type == 'b7':
                model_arch = "efficientnet-{}".format(args.model_type)
                if pretrained:
                    self._model = dt.model.efficientnet.EfficientNetLM.from_pretrained(model_arch)
                else:
                    self._model = dt.model.efficientnet.EfficientNetLM.from_name(model_arch)
        elif args.model_name == 'efficientnet_rw':
            if args.model_type == 'b0' or \
               args.model_type == 'b1' or \
               args.model_type == 'b2' or \
               args.model_type == 'b3' or \
               args.model_type == 'b4' or \
               args.model_type == 'b5' or \
               args.model_type == 'b6' or \
               args.model_type == 'b7':
                model_arch = "efficientnet_{}".format(args.model_type)
                self._model = dt.model.timm.create_model(model_arch, pretrained=pretrained)
        elif args.model_name == 'fairnas':
            if args.model_type == 'a':
                self._model = dt.model.fairnas.FairNasA()         # 8-gpu
        elif args.model_name == 'resnet_rw':
            #if dt.train.is_chief():
            #    dt.print_pp(dt.model.timm.list_models())
            if args.model_type == '34':
                self._model = dt.model.timm.create_model('resnet34', pretrained=pretrained)
            elif args.model_type == '50':
                self._model = dt.model.timm.create_model('resnet50', pretrained=pretrained)
        else:
            #if dt.train.is_chief():
            #    dt.print_pp(torchvision.models.__dict__)
            self._model = torchvision.models.__dict__[args.model_name](pretrained=pretrained)

        dt.info(dt.DC.TRAIN, "model {}, type {}, pretrained {}".format(args.model_name, args.model_type, args.pretrained))

        return True

    def post_model(self):
        args = self._ctx.args
        if dt.train.is_chief():
            dt.summary.summary_model_patch(self._model)
            dt.info(dt.DC.TRAIN, "\n{}".format(dt.summary.summary_model_fwd(self._model, (3, args.out_size, args.out_size), device='cpu')))
            dt.summary.summary_model_patch(self._model, patch_fn=dt.summary.patch_clear_dt)

    def build_optimizer(self):

        if self._ctx.optim == 'RMSpropRW':
            self._optimizer = dt.optimizer.RMSpropRW(self._model.parameters(), lr=dt.train.get_lr_val(),
                    alpha=self._ctx.alpha, eps=self._ctx.opt_eps,
                    momentum=self._ctx.momentum, weight_decay=self._ctx.weight_decay,
                    centered=False, decoupled_decay=False, lr_in_momentum=True)
        elif self._ctx.optim == 'RMSpropNA':
            self._optimizer = dt.optimizer.RMSpropNA(self._model.parameters(), lr=dt.train.get_lr_val(),
                    rho=self._ctx.alpha, eps=self._ctx.opt_eps,
                    momentum=self._ctx.momentum, weight_decay=self._ctx.weight_decay,
                    warmup=0)
        elif self._ctx.optim == 'SDG':
            self._optimizer = optim.SGD(self._model.parameters(), lr=dt.train.get_lr_val(),
                momentum=self._ctx.momentum, weight_decay=self._ctx.weight_decay)
        else:
            self._optimizer = None

        return True

# Train
with dt.ctx(optim=ARGS.optim, data_format=ARGS.data_format,
            lr_initial=ARGS.lr_initial, lr_minimal=ARGS.lr_minimal, lr_curve=ARGS.lr_curve):
    dt.train.train(args=ARGS, est_class=ImageNetEstimator, est_cfg=dt.Opt(),
                   batch_size=ARGS.batch_size, valid_size=ARGS.valid_size,
                   validate_ep=ARGS.validate_ep, max_ep=ARGS.max_ep,
                   model_dir=ARGS.model_dir, save_interval=ARGS.save_interval,
                   alpha=ARGS.alpha, beta1=ARGS.beta1, beta2=ARGS.beta2, opt_eps=ARGS.opt_eps,
                   momentum=ARGS.momentum, weight_decay=ARGS.weight_decay,
                   random_seed=dt.util.random_int(1, 999999), gpu0=ARGS.gpu0, valid_only=ARGS.valid_only)

#dt.util.datalink_close()

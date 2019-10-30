from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import horovod.torch as hvd

import numpy as np
import time


_use_cuda = False
_device = torch.device('cpu')
_device_index = 0
_device_count = 0
_global_step = None
_learning_rate = None


def init_device(opt):
    global _use_cuda
    global _device
    global _device_index
    global _device_count
    _use_cuda = not opt.no_cuda and torch.cuda.is_available()
    _device = torch.device('cuda' if _use_cuda else 'cpu')
    _device_index = hvd.local_rank() + opt.gpu0
    _device_count = torch.cuda.device_count()

def use_cuda():
    global _use_cuda
    return _use_cuda

def device():
    global _device
    return _device

def device_index():
    global _device_index
    return _device_index

def device_count():
    global _device_count
    return _device_count

def is_chief():
    return hvd.rank() == 0

def is_mp():
    return hvd.size() > 1

def global_step():
    global _global_step
    return _global_step

def global_step_inc():
    global _global_step
    _global_step += 1
    return _global_step

def init_global_step(opt):
    global _global_step
    _global_step = 0

def set_lr_val(lr):
    global _learning_rate
    _learning_rate = lr

def get_lr_val():
    global _learning_rate
    return _learning_rate

def init_learning_rate(opt):

    set_lr_val(opt.lr_initial)

    dt.info(dt.DC.TRAIN, 'Initialize learning rate: initial {}, minimal {}, curve {}'
                             .format(opt.lr_initial, opt.lr_minimal, opt.lr_curve))

def init_summary(opt):
    # summary writer
    opt.log_dir = opt.args.inst_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
    opt.summary_writer = SummaryWriter(opt.log_dir)
    dt.vis.set_default_writer(opt.summary_writer)

def optim_func(loss, **kwargs):
    opt = dt.Opt(kwargs)

    # default training options
    opt += dt.Opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, momentum=0.9, category='')

    dt.debug(dt.DC.TRAIN, "[OPTIM] {}, lr {}, beta1 {}, beta2 {}, momentum {}, category {}, deferred {}"
                                 .format(opt.optim, opt.lr, opt.beta1, opt.beta2, opt.momentum, opt.catetory, opt.deferred))

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_learning_rate(optimizer):
    lr = get_lr_val()
    adjust_learning_rate(optimizer, lr)

def dump_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

def train(**kwargs):

    opt = dt.Opt(kwargs) + dt.get_ctx()

    # Set default device settings
    opt += dt.Opt(gpu0=0)

    # Set default train mode
    opt += dt.Opt(is_training=True, is_eval=False, is_pred=False)

    # Learning rate
    opt += dt.Opt(lr_initial=0.001, lr_minimal=1e-6, lr_curve=[['*', 0.1, 10, 1]])

    # Default training options
    opt += dt.Opt(optim='SGD', beta1=0.9, beta2=0.99, momentum=0.9, weight_decay=5e-4, category='',
                  model_dir='asset/train', random_seed=12345, op_random_seed=12345,
                  max_ep=100000, summary_freq=16, summary_steps=100,
                  save_interval=600, max_keep=5, keep_interval=1000,
                  valid_metric=[], validate_ep=0, data_format=dt.dformat.DEFAULT)

    # Stats
    opt += dt.Opt(stats=dt.Opt(avg_loss=None, pri_metric_name=None, pri_metric=None, train_speed=None, valid_speed=None))

    if is_chief():
        dt.info(dt.DC.TRAIN, '[TRAIN] opt')
        dt.print_pp(dt.opt_to_dict(opt))

    # Initialize device
    init_device(opt)
    dt.info(dt.DC.TRAIN, '[HOROVOD] rank {}/{}, local {}'
                             .format(hvd.rank(), hvd.size(), hvd.local_rank()))
    dt.info(dt.DC.TRAIN, '[DEVICE] use_cuda {}, device {}, gpu {}/{}'
                             .format(use_cuda(), device(), device_index(), device_count()))

    # Initialize training variables
    init_global_step(opt)
    init_learning_rate(opt)
    init_summary(opt)

    #if opt.summary_freq > 0:
    #    opt.summary_steps = opt.data.ep_size // opt.summary_freq

    torch.manual_seed(opt.random_seed)
    if use_cuda():
        torch.cuda.set_device(device_index())

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    # Estimiator
    est = opt.est_class(opt, opt.est_cfg)
    est.build_estimator()

    # Local variables
    train_loader = est.data.train.loader
    valid_loader = est.data.valid.loader

    # Save graph
    images, labels = next(iter(train_loader))
    dt.vis.add_graph(est.model, images.to(est.device))
    dt.vis.add_images_grid('model/inputs', images)

    # Hooks
    train_hooks = dt.train.TrainCallGroup(opt)
    train_hooks.add(dt.train.LearningRateHook(opt, lr_val=get_lr_val(), lr_minimal=opt.lr_minimal, lr_curve=opt.lr_curve, optimizer=est.optimizer))
    for hook in est.train_hooks:
        train_hooks.add(hook)
    train_hooks.add(dt.train.TrainProgressHook(opt, every_n_steps=1))

    valid_hooks = dt.train.TrainCallGroup(opt)
    for hook in est.valid_hooks:
        valid_hooks.add(hook)
    valid_hooks.add(dt.train.ValidProgressHook(opt, every_n_steps=1))

    # Training
    est.pre_train()

    train_start = time.time()
    train_hooks.begin(train_start=train_start)
    valid_hooks.begin(train_start=train_start)

    for epoch in range(0, opt.max_ep):

        # Train
        est.model.train()
        opt.is_training = True
        opt.is_eval = False

        train_hooks.pre_epoch(step=global_step(), epoch=epoch,
                              epoch_size=est.data.train.num_batch,
                              batch_size=est.data.train.batch_size)
        dt.vis.add_scalar('train/epoch', epoch+1)
        dt.vis.add_scalar('train/lr', get_lr_val())

        for index, (images, labels) in enumerate(train_loader):
            size = len(images)
            train_hooks.pre_step(step=global_step(), epoch=epoch,
                                 index=index, size=size)

            images, labels = images.to(est.device), labels.to(est.device)

            logits = est.forward(images, opt.is_training)

            loss = est.criterion(logits, labels)

            est.optimizer.zero_grad()
            loss.backward()
            est.optimizer.step()

            metric = est.metric(logits, labels, opt.is_training)

            train_hooks.post_step(step=global_step(), epoch=epoch,
                                  index=index, size=size,
                                  images=images, labels=labels,
                                  logits=logits, loss=loss, metric=metric)

            global_step_inc()

        train_hooks.post_epoch(step=global_step(), epoch=epoch)

        # Validation
        if opt.validate_ep > 0 and (epoch+1) % opt.validate_ep == 0:

            est.model.eval()
            opt.is_training = False
            opt.is_eval = True

            valid_hooks.pre_epoch(step=global_step(), epoch=epoch,
                                  epoch_size=est.data.valid.num_batch,
                                  batch_size=est.data.valid.batch_size)

            with torch.no_grad():
                for index, (images, labels) in enumerate(valid_loader):
                    size = len(images)
                    valid_hooks.pre_step(step=global_step(), epoch=epoch,
                                         index=index, size=size)

                    images, labels = images.to(est.device), labels.to(est.device)

                    logits = est.forward(images, opt.is_training)

                    loss = est.criterion(logits, labels)

                    metric = est.metric(logits, labels, opt.is_training)

                    valid_hooks.post_step(step=global_step(), epoch=epoch,
                                          index=index, size=size,
                                          images=images, labels=labels,
                                          logits=logits, loss=loss, metric=metric)

            valid_hooks.post_epoch(step=global_step(), epoch=epoch)

        # End of epoch
        opt.summary_writer.flush()

    train_end = time.time()
    train_hooks.end(train_end=train_end)
    valid_hooks.end(train_end=train_end)
    est.post_train()

    # Save model
    #if (args.save_model):
    #    torch.save(est.model().state_dict(), "mnist_cnn.pt")

    opt.summary_writer.close()


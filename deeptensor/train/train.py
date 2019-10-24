from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import horovod.torch as hvd

import numpy as np
import time


_global_step = None
_learning_rate = None

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

def is_chief():
    return hvd.rank() == 0

def init_summary(opt):
    # summary writer
    opt.log_dir = opt.args.inst_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
    #opt.summary_writer = tf.summary.FileWriter(opt.log_dir)

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

    # Set default train mode
    opt += dt.Opt(is_training=True, is_eval=False, is_pred=False)

    # Learning rate
    opt += dt.Opt(lr_initial=0.001, lr_minimal=1e-6, lr_curve=[['*', 0.1, 10, 1]])

    # Default training options
    opt += dt.Opt(optim='MaxProp', beta1=0.9, beta2=0.99, momentum=0.9, category='',
                  model_dir='asset/train', random_seed=12345, op_random_seed=12345,
                  max_ep=100000, summary_freq=16, summary_steps=100,
                  save_interval=600, max_keep=5, keep_interval=1000,
                  valid_metric=[], validate_ep=0, data_format=dt.dformat.DEFAULT)

    # Stats
    opt += dt.Opt(stats=dt.Opt(avg_loss=None, avg_acc=None, train_speed=None, valid_speed=None))

    dt.info(dt.DC.TRAIN, '[TRAIN] opt')
    dt.print_pp(dt.opt_to_dict(opt))

    dt.info(dt.DC.TRAIN, '[HOROVOD] rank {}/{}, local {}'
                             .format(hvd.rank(), hvd.size(), hvd.local_rank()))

    # Initialize
    init_global_step(opt)
    init_learning_rate(opt)
    init_summary(opt)

    #if opt.summary_freq > 0:
    #    opt.summary_steps = opt.data.ep_size // opt.summary_freq

    torch.manual_seed(opt.random_seed)

    # Estimiator
    est = opt.est_class(opt, opt.est_cfg)
    est.build_estimator()

    # Local variables
    device = est.device
    train_loader = est.data.train.loader
    valid_loader = est.data.valid.loader
    model = est.model

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=get_lr_val(), momentum=opt.momentum)

    # Hooks
    train_hooks = dt.train.TrainCallGroup(opt)
    train_hooks.add(dt.train.LearningRateHook(opt, lr_val=get_lr_val(), lr_minimal=opt.lr_minimal, lr_curve=opt.lr_curve, optimizer=optimizer))
    train_hooks.add(dt.train.TrainProgressHook(opt, every_n_steps=1))

    valid_hooks = dt.train.TrainCallGroup(opt)
    valid_hooks.add(dt.train.ValidProgressHook(opt, every_n_steps=1))

    # Training
    est.pre_train()
    train_start = time.time()
    train_hooks.begin(train_start=train_start)
    valid_hooks.begin(train_start=train_start)

    for epoch in range(0, opt.max_ep):

        train_hooks.pre_epoch(step=global_step(), epoch=epoch,
                              epoch_size=est.data.train.num_batch,
                              batch_size=est.data.train.batch_size)
        model.train()
        opt.is_training = True
        opt.is_eval = False
        train_loss = None
        train_correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            train_hooks.pre_step(step=global_step(), epoch=epoch, batch=batch_idx)

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            train_loss = F.nll_loss(output, target)
            train_loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct = pred.eq(target.view_as(pred)).sum().item()

            train_hooks.post_step(step=global_step(), epoch=epoch,
                                  batch=batch_idx, batch_size=len(data),
                                  data=data, target=target,
                                  output=output, pred=pred,
                                  loss=train_loss.item(), correct=train_correct)

            global_step_inc()

        train_hooks.post_epoch(step=global_step(), epoch=epoch)

        if opt.validate_ep > 0 and (epoch+1) % opt.validate_ep == 0:
            valid_hooks.pre_epoch(step=global_step(), epoch=epoch,
                                  epoch_size=est.data.valid.num_batch,
                                  batch_size=est.data.valid.batch_size)
            model.eval()
            opt.is_training = False
            opt.is_eval = True
            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    valid_hooks.pre_step(step=global_step(), epoch=epoch, batch=batch_idx)

                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()

                    valid_hooks.post_step(step=global_step(), epoch=epoch,
                                          batch=batch_idx, batch_size=len(data),
                                          data=data, target=target,
                                          output=output, pred=pred,
                                          loss=test_loss, correct=test_correct)

            valid_hooks.post_epoch(step=global_step(), epoch=epoch,
                                   loss=test_loss, correct=test_correct)

    train_end = time.time()
    train_hooks.end(train_end=train_end)
    valid_hooks.end()
    est.post_train()

    # Save model
    #if (args.save_model):
    #    torch.save(est.model().state_dict(), "mnist_cnn.pt")


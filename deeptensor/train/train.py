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
from tqdm import tqdm


_global_step = None
_learning_rate = None
_lr_val = 0.1

def global_step():
    global _global_step
    return _global_step

def global_step_inc():
    global _global_step
    _global_step += 1
    return _global_step

def init_gs(opt):
    global _global_step
    _global_step = 0

def set_lr_val(lr):
    global _lr_val
    _lr_val = lr

def get_lr_val():
    global _lr_val
    return _lr_val

def init_lr(opt):
    global _learning_rate

    #_learning_rate = tf.placeholder_with_default(tf.constant(opt.lr_initial, tf.float32), [], name='learning_rate')
    set_lr_val(opt.lr_initial)

    dt.info(dt.DC.TRAIN, 'Initialize learning rate: initial {}, minimal {}, curve {}'
                             .format(opt.lr_initial, opt.lr_minimal, opt.lr_curve))

    # add learning rate summary
    #opt.lr = _learning_rate #* hvd.size()
    #tf.summary.scalar('learning_r', opt.lr)

def is_chief():
    return hvd.rank() == 0

def init_summary(opt):
    # summary writer
    opt.log_dir = opt.args.inst_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
    #opt.summary_writer = tf.summary.FileWriter(opt.log_dir)

def _close_tqdm(opt):
    if opt.tqdm is not None:
        opt.tqdm.close()
        opt.tqdm = None

def optim_func(loss, **kwargs):
    opt = dt.Opt(kwargs)

    # default training options
    opt += dt.Opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, momentum=0.9, category='')

    dt.debug(dt.DC.TRAIN, "[OPTIM] {}, lr {}, beta1 {}, beta2 {}, momentum {}, category {}, deferred {}"
                                 .format(opt.optim, opt.lr, opt.beta1, opt.beta2, opt.momentum, opt.catetory, opt.deferred))

def train(**kwargs):

    opt = dt.Opt(kwargs) + dt.get_ctx()

    # set default train mode
    opt += dt.Opt(is_training=True, is_eval=False, is_pred=False)

    # learning rate
    opt += dt.Opt(lr_initial=0.001, lr_minimal=1e-6, lr_curve=[[0.1, 10, 1]])

    # default training options
    opt += dt.Opt(optim='MaxProp', beta1=0.9, beta2=0.99, momentum=0.9, category='',
                  model_dir='asset/train', random_seed=12345, op_random_seed=12345,
                  max_ep=100000, summary_freq=16, summary_steps=100,
                  save_interval=600, max_keep=5, keep_interval=1000,
                  valid_metric=[], validate_ep=0, data_format=dt.dformat.DEFAULT,
                  tqdm=None)

    # stats
    opt += dt.Opt(stats=dt.Opt(avg_loss=None, avg_acc=None, train_speed=None))

    dt.info(dt.DC.TRAIN, '[TRAIN] opt')
    dt.print_pp(dt.opt_to_dict(opt))

    dt.info(dt.DC.TRAIN, '[HOROVOD] rank {}/{}, local {}'
                             .format(hvd.rank(), hvd.size(), hvd.local_rank()))

    init_gs(opt)
    init_lr(opt)
    init_summary(opt)

    #if opt.summary_freq > 0:
    #    opt.summary_steps = opt.data.ep_size // opt.summary_freq

    torch.manual_seed(opt.random_seed)

    est = opt.est_class(opt, opt.est_cfg)
    est.build_estimator()

    device = est.device
    train_loader = est.data.train.loader
    valid_loader = est.data.valid.loader
    model = est.model

    optimizer = optim.SGD(model.parameters(), lr=opt.lr_initial, momentum=opt.momentum)

    # Hooks
    train_hooks = dt.train.TrainCallGroup(opt)
    t_hook = dt.train.TrainProgressHook(opt, every_n_steps=1, est=est, optimizer=optimizer)
    t_hook_idx = train_hooks.add(t_hook)

    valid_hooks = dt.train.TrainCallGroup(opt)
    v_hook = dt.train.ValidProgressHook(opt, every_n_steps=1, est=est)
    v_hook_idx = valid_hooks.add(v_hook)

    est.pre_train()
    train_start = time.time()

    #bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
    train_hooks.begin(train_start=train_start)
    valid_hooks.begin(train_start=train_start)
    for epoch in range(0, 10):

        train_hooks.pre_epoch(step=global_step(), epoch=epoch,
                              train_loader=train_loader, valid_loader=valid_loader,
                              epoch_size=est.data.train.num_batch,
                              batch_size=est.data.train.batch_size)
        model.train()
        train_loss = None
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            train_hooks.pre_step(step=global_step(), epoch=epoch, batch=batch_idx,
                                 train_loader=train_loader, valid_loader=valid_loader)

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            train_loss = F.nll_loss(output, target)
            train_loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct = pred.eq(target.view_as(pred)).sum().item()

            train_hooks.post_step(step=global_step(), epoch=epoch, batch=batch_idx,
                                  train_loader=train_loader, valid_loader=valid_loader,
                                  data=data, target=target,
                                  output=output, loss=train_loss.item(), correct=train_correct)

            #if (batch_idx+1) % 10 == 0:
            #    print('Train Epoch: {} {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, (batch_idx+1), (batch_idx+1) * len(data), len(train_loader.dataset),
            #        100. * (batch_idx+1) / len(train_loader), loss.item()))

            global_step_inc()

        train_hooks.post_epoch(step=global_step(), epoch=epoch,
                               train_loader=train_loader, valid_loader=valid_loader)

        if opt.validate_ep > 0 and (epoch+1) % opt.validate_ep == 0:
            valid_hooks.pre_epoch(step=global_step(), epoch=epoch,
                                  train_loader=train_loader, valid_loader=valid_loader,
                                  epoch_size=est.data.valid.num_batch,
                                  batch_size=est.data.valid.batch_size)
            model.eval()
            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    valid_hooks.pre_step(step=global_step(), epoch=epoch, batch=batch_idx,
                                         train_loader=train_loader, valid_loader=valid_loader)

                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()

                    valid_hooks.post_step(step=global_step(), epoch=epoch, batch=batch_idx,
                                          train_loader=train_loader, valid_loader=valid_loader,
                                          data=data, target=target,
                                          output=output, pred=pred,
                                          loss=test_loss, correct=test_correct)

            #test_loss /= len(valid_loader.dataset)

            valid_hooks.post_epoch(step=global_step(), epoch=epoch,
                                   train_loader=train_loader, valid_loader=valid_loader,
                                   loss=test_loss, correct=test_correct)

            #print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            #    test_loss, test_correct, len(valid_loader.dataset),
            #    100. * test_correct / len(valid_loader.dataset)))

    train_end = time.time()
    train_hooks.end(train_end=train_end)
    valid_hooks.end()
    est.post_train()
    print(time.strftime("%H:%M:%S", time.gmtime(train_end - train_start)))

    #if (args.save_model):
    #    torch.save(est.model().state_dict(), "mnist_cnn.pt")


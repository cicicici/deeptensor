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

def init_library():
    # Horovod: initialize library.
    hvd.init()
    torch.backends.cudnn.benchmark = True

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

def chief_rank():
    return 0

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

def set_global_step(step):
    global _global_step
    _global_step = step

def set_lr_val(lr):
    global _learning_rate
    _learning_rate = lr

def get_lr_val():
    global _learning_rate
    return _learning_rate

def set_lr_val_mp(lr):
    global _learning_rate
    _learning_rate = lr * hvd.size()

def get_lr_val_base():
    global _learning_rate
    return _learning_rate / hvd.size()

def mp_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def mp_broadcast(params):
    tensor_params = {}
    for key, value in params.items():
        tensor_params[key] = torch.tensor(value)

    hvd.broadcast_parameters(tensor_params, root_rank=chief_rank())

    sync_params = {}
    for key, tensor in tensor_params.items():
        sync_params[key] = tensor.item()

    return sync_params

def init_learning_rate(opt):
    # Horovod: scale learning rate by the number of GPUs.
    set_lr_val_mp(opt.lr_initial)

    dt.info(dt.DC.TRAIN, 'Initialize learning rate: initial {} * {}, minimal {}, curve {}'
                             .format(opt.lr_initial, hvd.size(), opt.lr_minimal, opt.lr_curve))

def init_summary(opt):
    # summary writer
    opt.log_dir = opt.args.inst_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
    opt.summary_writer = SummaryWriter(opt.log_dir)
    dt.vis.set_default_writer(opt.summary_writer)

def init_saver(opt):
    # checkpoint
    opt.saver = dt.Opt(model_latest = opt.args.inst_dir + '/model_latest.pt',
                       optimizer_latest = opt.args.inst_dir + '/optimizer_latest.pt',
                       model_best = opt.args.inst_dir + '/model_best.pt',
                       optimizer_best = opt.args.inst_dir + '/optimizer_best.pt')

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
                  model_dir='asset/train', random_seed=12345, max_ep=100000,
                  save_interval=1, max_keep=5, validate_ep=0, data_format=dt.dformat.DEFAULT)

    # Default horovod options
    opt += dt.Opt(fp16_allreduce=False)

    # Stats
    opt += dt.Opt(stats=dt.Opt(avg_loss=None, train_metric_name=None, train_metric=None,
                               valid_loss=0, valid_metric_name='', valid_metric=0, valid_metric_max=None,
                               train_speed=0, valid_speed=0))

    # Saver
    opt += dt.Opt(epoch_done=-1)

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
    init_saver(opt)

    if is_chief():
        dt.info(dt.DC.TRAIN, '[TRAIN] opt')
        dt.print_pp(dt.opt_to_dict(opt))

    torch.manual_seed(opt.random_seed)
    if use_cuda():
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(device_index())

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    # Estimiator
    est = opt.est_class(opt, opt.est_cfg)
    est.build_estimator()

    # Load checkpoint
    sync_params = {'epoch_done': opt.epoch_done, 'global_step': global_step()}
    if is_chief():
        model_params = dt.model.load(est.model, opt.saver.model_latest)
        optimizer_params = dt.optimizer.load(est.optimizer, opt.saver.optimizer_latest)
        if optimizer_params:
            sync_params['epoch_done'] = optimizer_params.epoch
            sync_params['global_step'] = optimizer_params.step
            #opt.stats = optimizer_params.stats

    sync_params = mp_broadcast(sync_params)
    opt.epoch_done = int(sync_params['epoch_done'])
    set_global_step(int(sync_params['global_step']))
    dt.trace(dt.DC.TRAIN, '[CHECKPOINT] epoch_done {}, global_step {}'.format(opt.epoch_done, global_step()))

    if use_cuda():
        # Move model to GPU.
        est.model.cuda()

    # Make sure learning rate is up to date
    update_learning_rate(est.optimizer)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(est.model.state_dict(), root_rank=chief_rank())
    hvd.broadcast_optimizer_state(est.optimizer, root_rank=chief_rank())

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if opt.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    est.optimizer = hvd.DistributedOptimizer(est.optimizer,
                                             named_parameters=est.model.named_parameters(),
                                             compression=compression)

    # Local variables
    train_loader = est.data.train.loader
    valid_loader = est.data.valid.loader

    # Hooks
    train_hooks = dt.train.TrainCallGroup(opt)
    if not opt.valid_only:
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
    if not opt.valid_only:
        train_hooks.begin(train_start=train_start)
    valid_hooks.begin(train_start=train_start)

    for epoch in range(0, opt.max_ep):
        dryrun = (epoch <= opt.epoch_done)

        # Train
        if not opt.valid_only:
            est.model.train()
            opt.is_training = True
            opt.is_eval = False
            if est.data.train.sampler is not None:
                # Horovod: set epoch to sampler for shuffling.
                est.data.train.sampler.set_epoch(epoch)

            train_hooks.pre_epoch(step=global_step(), epoch=epoch,
                                  epoch_size=est.data.train.num_batch,
                                  batch_size=est.data.train.batch_size,
                                  dryrun=dryrun)

            dt.vis.add_scalar('train/epoch', epoch+1)
            dt.vis.add_scalar('train/lr', get_lr_val())

            for index, (images, labels) in enumerate(train_loader):
                size = len(images)
                train_hooks.pre_step(step=global_step(), epoch=epoch,
                                     index=index, size=size,
                                     dryrun=dryrun)

                if not dryrun:
                    if use_cuda():
                        images, labels = images.cuda(), labels.cuda()

                    # Save graph
                    if global_step() == 0:
                        #images, labels = next(iter(train_loader))
                        dt.vis.add_graph(est.model, images.to(est.device))
                        dt.vis.add_images_grid('model/inputs', images)

                    logits = est.forward(images, opt.is_training)

                    loss = est.criterion(logits, labels)

                    est.optimizer.zero_grad()
                    loss.backward()
                    est.optimizer.step()

                    metric = est.metric(logits, labels, opt.is_training)
                else:
                    logits = None
                    loss = None
                    metric = None

                train_hooks.post_step(step=global_step(), epoch=epoch,
                                      index=index, size=size,
                                      images=images, labels=labels,
                                      logits=logits, loss=loss, metric=metric,
                                      dryrun=dryrun)

                if not dryrun:
                    global_step_inc()

            train_hooks.post_epoch(step=global_step(), epoch=epoch,
                                   dryrun=dryrun)

        # Skip validation for previous done epochs
        if dryrun:
            continue

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

                    if use_cuda():
                        images, labels = images.cuda(), labels.cuda()

                    logits = est.forward(images, opt.is_training)

                    loss = est.criterion(logits, labels)

                    metric = est.metric(logits, labels, opt.is_training)

                    valid_hooks.post_step(step=global_step(), epoch=epoch,
                                          index=index, size=size,
                                          images=images, labels=labels,
                                          logits=logits, loss=loss, metric=metric)

            valid_hooks.post_epoch(step=global_step(), epoch=epoch)

        if opt.valid_only:
            # Only need 1 epoch for valid
            break

        if is_chief():
            # Save checkpoint
            if opt.save_interval > 0 and (epoch+1) % opt.save_interval == 0:
                dt.model.save(est.model, opt.saver.model_latest,
                              valid_loss=opt.stats.valid_loss,
                              valid_metric_name=opt.stats.valid_metric_name,
                              valid_metric=opt.stats.valid_metric)
                dt.optimizer.save(est.optimizer, opt.saver.optimizer_latest,
                                  step=global_step(), epoch=epoch,
                                  lr_val_base=get_lr_val_base(), stats=opt.stats)

            # Save best model
            if (opt.stats.valid_metric_max is None or \
                opt.stats.valid_metric > opt.stats.valid_metric_max):
                dt.model.save(est.model, opt.saver.model_best,
                              valid_loss=opt.stats.valid_loss,
                              valid_metric_name=opt.stats.valid_metric_name,
                              valid_metric=opt.stats.valid_metric,
                              step=global_step(), epoch=epoch,
                              lr_val_base=get_lr_val_base(), stats=opt.stats)
                dt.optimizer.save(est.optimizer, opt.saver.optimizer_best,
                                  step=global_step(), epoch=epoch,
                                  lr_val_base=get_lr_val_base(), stats=opt.stats,
                                  optim=opt.optim, momentum=opt.momentum, weight_decay=opt.weight_decay,
                                  lr_initial=opt.lr_initial, lr_minimal=opt.lr_minimal, lr_curve=opt.lr_curve)
                opt.stats.valid_metric_max = opt.stats.valid_metric

        # End of epoch
        opt.summary_writer.flush()

    train_end = time.time()
    if not opt.valid_only:
        train_hooks.end(train_end=train_end)
    valid_hooks.end(train_end=train_end)
    est.post_train()

    # Save model
    #if (args.save_model):
    #    torch.save(est.model().state_dict(), "mnist_cnn.pt")

    opt.summary_writer.close()

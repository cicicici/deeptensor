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

import random
import numpy as np
import time


_mono_step = 0

def mono_step():
    return _mono_step

def mono_step_inc(step=1):
    global _mono_step
    _mono_step += step
    return _mono_step

def set_mono_step(step):
    global _mono_step
    _mono_step = step

def init_library():
    # Horovod: initialize library.
    hvd.init()
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.deterministic = True

def chief_rank():
    return 0

def is_chief():
    return hvd.rank() == 0

def is_mp():
    return hvd.size() > 1

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

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def dump_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

class Trainer(object):

    def __init__(self, **kwargs):
        self._ctx = dt.Opt(kwargs) + dt.get_ctx()

        self._use_cuda = False
        self._device = torch.device('cpu')
        self._device_index = 0
        self._device_count = 0

        self._global_step = None
        self._learning_rate = None

    @property
    def ctx(self):
        return self._ctx

    def init_device(self):
        self._use_cuda = not self.ctx.no_cuda and torch.cuda.is_available()
        self._device = torch.device('cuda' if self._use_cuda else 'cpu')
        self._device_index = hvd.local_rank() + self.ctx.gpu0
        self._device_count = torch.cuda.device_count()

    @property
    def use_cuda(self):
        return self._use_cuda

    @property
    def device(self):
        return self._device

    @property
    def device_index(self):
        return self._device_index

    @property
    def device_count(self):
        return self._device_count

    @property
    def global_step(self):
        return self._global_step

    def init_global_step(self):
        self._global_step = 0

    def global_step_inc(self, step=1):
        mono_step_inc(step)

        self._global_step += step
        return self._global_step

    def set_global_step(self, step):
        self._global_step = step

    def set_lr_val(self, lr):
        self._learning_rate = lr

    def get_lr_val(self):
        return self._learning_rate

    def update_learning_rate(self, optimizer):
        lr = self.get_lr_val()
        adjust_learning_rate(optimizer, lr)

    def set_lr_val_mp(self, lr):
        self._learning_rate = lr * hvd.size()

    def get_lr_val_base(self):
        return self._learning_rate / hvd.size()

    def init_learning_rate(self):
        # Horovod: scale learning rate by the number of GPUs.
        self.set_lr_val_mp(self.ctx.lr_initial)

        dt.info(dt.DC.TRAIN, 'Initialize learning rate: initial {} * {}, minimal {}, curve {}'
                                 .format(self.ctx.lr_initial, hvd.size(), self.ctx.lr_minimal, self.ctx.lr_curve))

    def init_summary(self):
        # summary writer
        self._log_dir = self.ctx.args.inst_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
        self._summary_writer = SummaryWriter(self._log_dir)
        dt.vis.set_default_writer(self._summary_writer)

    def init_saver(self):
        # checkpoint
        self._saver = dt.Opt(model_latest = self.ctx.args.inst_dir + '/model_latest.pt',
                             optimizer_latest = self.ctx.args.inst_dir + '/optimizer_latest.pt',
                             model_best = self.ctx.args.inst_dir + '/model_best.pt',
                             optimizer_best = self.ctx.args.inst_dir + '/optimizer_best.pt')

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def init(self, **kwargs):

        opt = dt.Opt(kwargs) + dt.get_ctx() + self._ctx

        # Set default device settings
        opt += dt.Opt(gpu0=0)

        # Set default train mode
        opt += dt.Opt(is_training=True, is_eval=False, is_pred=False)

        # Learning rate
        opt += dt.Opt(lr_initial=0.001, lr_minimal=1e-6, lr_curve=[['*', 0.1, 10, 1]])

        # Default training options
        opt += dt.Opt(optim='SGD', alpha=0.9, beta1=0.9, beta2=0.99, opt_eps=1e-6, momentum=0.9, weight_decay=5e-4,
                      model_dir='asset/train', random_seed=0, max_ep=100000,
                      save_interval=1, validate_ep=1, data_format=dt.dformat.DEFAULT)

        # Default horovod options
        opt += dt.Opt(fp16_allreduce=False)

        # Stats
        opt += dt.Opt(stats=dt.Opt(avg_loss=None, train_metric_name=None, train_metric=None,
                                   valid_loss=0, valid_metric_name='', valid_metric=0, valid_metric_max=None,
                                   train_speed=0, valid_speed=0))

        # Saver
        opt += dt.Opt(epoch_done=-1)

        # Update ctx
        self._ctx = opt

        # Initialize device
        self.init_device()
        dt.info(dt.DC.TRAIN, '[HOROVOD] rank {}/{}, local {}'
                                 .format(hvd.rank(), hvd.size(), hvd.local_rank()))
        dt.info(dt.DC.TRAIN, '[DEVICE] use_cuda {}, device {}, gpu {}/{}, random_seed {}'
                                 .format(self.use_cuda, self.device, self.device_index, self.device_count, self._ctx.random_seed))

        if is_chief():
            dt.info(dt.DC.TRAIN, '[TRAIN] ctx')
            dt.print_pp(dt.opt_to_dict(self._ctx))

        # Initialize training variables
        self.init_global_step()
        self.init_learning_rate()
        self.init_summary()
        self.init_saver()

        if self._ctx.random_seed != 0:
            self.set_random_seed(self._ctx.random_seed)

        if self.use_cuda:
            # Horovod: pin GPU to local rank.
            torch.cuda.set_device(self.device_index)

        # Horovod: limit # of CPU threads to be used per worker.
        torch.set_num_threads(1)

    def bind_estimator(self, est_class):

        # Estimiator
        est = est_class(self._ctx)
        est.bind_trainer(self)
        est.build_estimator()

        # Load checkpoint
        sync_params = {'epoch_done': self._ctx.epoch_done, 'global_step': self.global_step}
        if is_chief():
            model_params = dt.model.load(est.model, self._saver.model_latest)
            optimizer_params = dt.optimizer.load(est.optimizer, self._saver.optimizer_latest)
            if optimizer_params:
                sync_params['epoch_done'] = optimizer_params.epoch
                sync_params['global_step'] = optimizer_params.step
                #self._ctx.stats = optimizer_params.stats

        sync_params = mp_broadcast(sync_params)
        self._ctx.epoch_done = int(sync_params['epoch_done'])
        self.set_global_step(int(sync_params['global_step']))
        mono_step_inc(self.global_step)
        dt.trace(dt.DC.TRAIN, '[CHECKPOINT] epoch_done {}, global_step {}, mono_step {}'.format(
            self._ctx.epoch_done, self.global_step, mono_step()))

        if self.use_cuda:
            # Move model to GPU.
            est.model.cuda()

        # Make sure learning rate is up to date
        self.update_learning_rate(est.optimizer)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(est.model.state_dict(), root_rank=chief_rank())
        hvd.broadcast_optimizer_state(est.optimizer, root_rank=chief_rank())

        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if self._ctx.fp16_allreduce else hvd.Compression.none

        # Horovod: wrap optimizer with DistributedOptimizer.
        est.optimizer = hvd.DistributedOptimizer(est.optimizer,
                                                 named_parameters=est.model.named_parameters(),
                                                 compression=compression)
        self._est = est

    def train(self):

        ctx = self._ctx
        est = self._est

        # Local variables
        train_loader = est.data.train.loader
        valid_loader = est.data.valid.loader

        # Hooks
        train_hooks = dt.train.TrainCallGroup(ctx)
        if not ctx.valid_only:
            train_hooks.add(dt.train.LearningRateHook(ctx, self, lr_val=self.get_lr_val(), lr_minimal=ctx.lr_minimal, lr_curve=ctx.lr_curve, optimizer=est.optimizer))
            for hook in est.train_hooks:
                train_hooks.add(hook)
            train_hooks.add(dt.train.TrainProgressHook(ctx, self, every_n_steps=1))

        valid_hooks = dt.train.TrainCallGroup(ctx)
        for hook in est.valid_hooks:
            valid_hooks.add(hook)
        valid_hooks.add(dt.train.ValidProgressHook(ctx, self, every_n_steps=1))

        # Training
        est.pre_train()

        train_start = time.time()
        if not ctx.valid_only:
            train_hooks.begin(train_start=train_start)
        valid_hooks.begin(train_start=train_start)

        for epoch in range(0, ctx.max_ep):
            dryrun = (epoch <= ctx.epoch_done)

            # Train
            if not ctx.valid_only:
                est.model.train()
                ctx.is_training = True
                ctx.is_eval = False
                if est.data.train.sampler is not None:
                    # Horovod: set epoch to sampler for shuffling.
                    est.data.train.sampler.set_epoch(epoch)

                train_hooks.pre_epoch(step=self.global_step, epoch=epoch,
                                      epoch_size=est.data.train.num_batch,
                                      batch_size=est.data.train.batch_size,
                                      dryrun=dryrun)

                dt.vis.add_scalar('train/epoch', epoch+1)
                dt.vis.add_scalar('train/lr', self.get_lr_val())

                train_it = iter(train_loader)
                for index in range(len(train_loader)):
                    images, labels = None, None
                    size = 0
                    if not dryrun:
                        images, labels = next(train_it)
                        size = len(images)

                    train_hooks.pre_step(step=self.global_step, epoch=epoch,
                                         index=index, size=size,
                                         images=images, labels=labels,
                                         dryrun=dryrun)

                    if not dryrun:
                        if self.use_cuda:
                            images, labels = images.cuda(), labels.cuda()

                        # Save graph
                        if self.global_step == 0:
                            #images, labels = next(iter(train_loader))
                            dt.vis.add_graph(est.model, images.to(self.device))
                            dt.vis.add_images_grid('model/inputs', images)

                        logits = est.forward(images, ctx.is_training)

                        loss = est.criterion(logits, labels)

                        est.optimizer.zero_grad()
                        loss.backward()
                        est.optimizer.step()

                        metric = est.metric(logits, labels, ctx.is_training)
                    else:
                        logits = None
                        loss = None
                        metric = None

                    train_hooks.post_step(step=self.global_step, epoch=epoch,
                                          index=index, size=size,
                                          images=images, labels=labels,
                                          logits=logits, loss=loss, metric=metric,
                                          dryrun=dryrun)

                    if not dryrun:
                        self.global_step_inc()

                train_hooks.post_epoch(step=self.global_step, epoch=epoch,
                                       dryrun=dryrun)

            # Skip validation for previous done epochs
            if dryrun:
                continue

            # Validation
            if ctx.validate_ep > 0 and (epoch+1) % ctx.validate_ep == 0:

                est.model.eval()
                ctx.is_training = False
                ctx.is_eval = True

                valid_hooks.pre_epoch(step=self.global_step, epoch=epoch,
                                      epoch_size=est.data.valid.num_batch,
                                      batch_size=est.data.valid.batch_size)

                with torch.no_grad():
                    for index, (images, labels) in enumerate(valid_loader):
                        size = len(images)
                        valid_hooks.pre_step(step=self.global_step, epoch=epoch,
                                             index=index, size=size)

                        if self.use_cuda:
                            images, labels = images.cuda(), labels.cuda()

                        logits = est.forward(images, ctx.is_training)

                        loss = est.criterion(logits, labels)

                        metric = est.metric(logits, labels, ctx.is_training)

                        valid_hooks.post_step(step=self.global_step, epoch=epoch,
                                              index=index, size=size,
                                              images=images, labels=labels,
                                              logits=logits, loss=loss, metric=metric)

                valid_hooks.post_epoch(step=self.global_step, epoch=epoch)

            if ctx.valid_only:
                # Only need 1 epoch for valid
                break

            if is_chief():
                # Save checkpoint
                if ctx.save_interval > 0 and (epoch+1) % ctx.save_interval == 0:
                    dt.model.save(est.model, self._saver.model_latest,
                                  valid_loss=ctx.stats.valid_loss,
                                  valid_metric_name=ctx.stats.valid_metric_name,
                                  valid_metric=ctx.stats.valid_metric)
                    dt.optimizer.save(est.optimizer, self._saver.optimizer_latest,
                                      step=self.global_step, epoch=epoch,
                                      lr_val_base=self.get_lr_val_base(), stats=ctx.stats)

                # Save best model
                if (ctx.stats.valid_metric_max is None or \
                    ctx.stats.valid_metric > ctx.stats.valid_metric_max):
                    dt.model.save(est.model, self._saver.model_best,
                                  valid_loss=ctx.stats.valid_loss,
                                  valid_metric_name=ctx.stats.valid_metric_name,
                                  valid_metric=ctx.stats.valid_metric,
                                  step=self.global_step, epoch=epoch,
                                  lr_val_base=self.get_lr_val_base(), stats=ctx.stats)
                    dt.optimizer.save(est.optimizer, self._saver.optimizer_best,
                                      step=self.global_step, epoch=epoch,
                                      lr_val_base=self.get_lr_val_base(), stats=ctx.stats,
                                      optim=ctx.optim, momentum=ctx.momentum, weight_decay=ctx.weight_decay,
                                      lr_initial=ctx.lr_initial, lr_minimal=ctx.lr_minimal, lr_curve=ctx.lr_curve)
                    ctx.stats.valid_metric_max = ctx.stats.valid_metric

            # End of epoch
            self._summary_writer.flush()

        train_end = time.time()
        if not ctx.valid_only:
            train_hooks.end(train_end=train_end)
        valid_hooks.end(train_end=train_end)
        est.post_train()

        # Save model
        #if (args.save_model):
        #    torch.save(est.model().state_dict(), "mnist_cnn.pt")

        self._summary_writer.close()

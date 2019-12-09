from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import horovod.torch as hvd

from tqdm import tqdm
import numpy as np
import time
import copy


class TrainHook(dt.util.Callback):

    def __init__(self, ctx, **kwargs):
        super(TrainHook, self).__init__(ctx, **kwargs)
        self._every_n_steps = self._ctx.every_n_steps

    def begin(self, **kwargs):
        pass

    def pre_epoch(self, **kwargs):
        #step = kwargs['step']
        #epoch = kwargs['epoch']
        return None

    def pre_step(self, **kwargs):
        #step = kwargs['step']
        #epoch = kwargs['epoch']
        #batch = kwargs['batch']
        return None

    def post_step(self, **kwargs):
        #step = kwargs['step']
        #epoch = kwargs['epoch']
        #batch = kwargs['batch']
        return None

    def post_epoch(self, **kwargs):
        #step = kwargs['step']
        #epoch = kwargs['epoch']
        return None

    def end(self, **kwargs):
        pass

class TrainProgressHook(TrainHook):

    def begin(self, **kwargs):
        pass

    def pre_epoch(self, **kwargs):
        self._epoch_size = kwargs['epoch_size']
        dryrun = kwargs['dryrun']

        if dryrun:
            return None

        self._epoch_start = time.time()
        self._step_total = 0
        self._num_total = 0
        if dt.train.is_chief():
            self._tqdm = tqdm(total=self._epoch_size, initial=0, desc='train', ncols=80, unit='b', leave=False)

        return None

    def post_step(self, **kwargs):
        size = kwargs['size']
        loss = kwargs['loss']
        metric = kwargs['metric']
        dryrun = kwargs['dryrun']

        if dryrun:
            return None

        self._step_total += 1
        self._num_total += size

        # loss history update
        loss_val = loss.item()
        if not np.isnan(loss_val) and not np.isinf(loss_val):
             if self._ctx.stats.avg_loss is None:
                self._ctx.stats.avg_loss = loss_val
             else:
                self._ctx.stats.avg_loss = self._ctx.stats.avg_loss * 0.9 + loss_val * 0.1

        # primary metric history update
        if self._ctx.stats.train_metric_name is None:
            self._ctx.stats.train_metric_name = metric[0].name
        if self._ctx.stats.train_metric is None:
            self._ctx.stats.train_metric = metric[0].tensor.item()
        else:
            self._ctx.stats.train_metric = self._ctx.stats.train_metric * 0.9 + metric[0].tensor.item() * 0.1

        if dt.train.is_chief():
            self._tqdm.update(1)

        return None

    def post_epoch(self, **kwargs):
        dryrun = kwargs['dryrun']

        if dryrun:
            return None

        elapsed_time = time.time() - self._epoch_start

        if dt.train.is_chief():
            self._tqdm.close()
            self._tqdm = None
            self._ctx.stats.train_speed = self._num_total / elapsed_time * hvd.size()
        else:
            self._ctx.stats.train_speed = self._num_total / elapsed_time

        dt.vis.add_scalar('meter/img/sec', self._ctx.stats.train_speed)
        dt.vis.add_scalar('meter/step/sec', self._step_total / elapsed_time)
        dt.vis.add_scalar('meter/time/epoch', elapsed_time)

        return None


class ValidProgressHook(TrainHook):

    def begin(self, **kwargs):
        self._train_start = kwargs['train_start']
        pass

    def pre_epoch(self, **kwargs):
        self._epoch_size = kwargs['epoch_size']

        self._epoch_start = time.time()
        self._num_total = 0
        self._loss_total = 0
        self._metric_total = [0, 0]
        self._metric_name = [None, None]
        if dt.train.is_chief():
            self._tqdm = tqdm(total=self._epoch_size, initial=0, desc='valid', ncols=80, unit='b', leave=False)

        return None

    def post_step(self, **kwargs):
        size = kwargs['size']
        loss = kwargs['loss']
        metric = kwargs['metric']

        self._num_total += size
        self._loss_total += loss.item() * size

        for i, val in enumerate(metric):
            if self._metric_name[i] is None:
                self._metric_name[i] = metric[i].name
            self._metric_total[i] += metric[i].tensor.mul(size)

        if dt.train.is_chief():
            self._tqdm.update(1)

        return None

    def post_epoch(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']

        now_time = time.time()

        if dt.train.is_chief():
            self._tqdm.close()
            self._tqdm = None
            self._ctx.stats.valid_speed = self._num_total / (now_time - self._epoch_start) * hvd.size()
        else:
            self._ctx.stats.valid_speed = self._num_total / (now_time - self._epoch_start)

        self._ctx.stats.valid_loss = dt.train.mp_average(self._loss_total/self._num_total, 'epoch_valid_loss')
        #dt.trace(dt.DC.TRAIN, '[EPOCH {}] local {}, avg {}'.format(epoch, self._loss_total/self._num_total, self._ctx.stats.valid_loss))
        if self._ctx.valid_only:
            train_avg_loss = 0
            train_metric = 0
        else:
            train_avg_loss = dt.train.mp_average(self._ctx.stats.avg_loss, 'epoch_avg_loss')
            train_metric = dt.train.mp_average(self._ctx.stats.train_metric, 'epoch_train_metric')

        dt.vis.add_scalar('valid/image/s', self._ctx.stats.valid_speed)
        dt.vis.add_scalar('valid/avg_loss', train_avg_loss)
        dt.vis.add_scalar('valid/avg_{}'.format(self._ctx.stats.train_metric_name), train_metric)
        dt.vis.add_scalar('valid/loss', self._ctx.stats.valid_loss)
        for i, val in enumerate(self._metric_name):
            if self._metric_name[i] is not None:
                self._metric_total[i].div_(self._num_total)
                self._metric_total[i] = hvd.allreduce(self._metric_total[i], name='epoch_metric_total_%d' % i)
                dt.vis.summary_tensor('metric/{}'.format(self._metric_name[i]), self._metric_total[i])
        self._ctx.stats.valid_metric_name = self._metric_name[0]
        self._ctx.stats.valid_metric = self._metric_total[0].item()

        # Horovod: print output only on first rank.
        if dt.train.is_chief():
            dt.info(dt.DC.TRAIN, '%s Epoch[%03d:lr=%.6f:gs=%06d] train (loss %s, %s %s), valid (loss %s, %s %s, %s %s), %.3f img/s' %
                                     (time.strftime("%H:%M:%S", time.gmtime(now_time - self._train_start)),
                                     (epoch+1), dt.train.get_lr_val(), step,
                                     "{:.6f}".format(train_avg_loss), self._ctx.stats.train_metric_name, "{:.6f}".format(train_metric),
                                     "{:.6f}".format(self._ctx.stats.valid_loss),
                                     self._metric_name[0], "{:.6f}".format(self._metric_total[0].item()),
                                     self._metric_name[1], "{:.6f}".format(self._metric_total[1].item()),
                                     self._ctx.stats.train_speed))
        return None


class LearningRateHook(TrainHook):

    def __init__(self, ctx, **kwargs):
        super(LearningRateHook, self).__init__(ctx, **kwargs)

        self._lr_val = kwargs['lr_val']
        self._lr_minimal = kwargs['lr_minimal']
        # [[op_0, factor_0, epoch_0, updates_0], ... [op_n, factor_n, epoch_n, update_n]]
        self._lr_curve = copy.deepcopy(kwargs['lr_curve'])
        self._optimizer = kwargs['optimizer']

    def begin(self, **kwargs):
        self._step_elapsed = 0
        self._epoch_window = self._lr_curve[0][2]
        pass

    def pre_epoch(self, **kwargs):
        self._epoch_size = kwargs['epoch_size']

    def pre_step(self, **kwargs):
        #step = kwargs['step']
        #epoch = kwargs['epoch']
        #index = kwargs['index']

        epoch_elapsed = self._step_elapsed / self._epoch_size

        if self._epoch_window > 0 and epoch_elapsed >= self._epoch_window:
            #dt.trace(dt.DC.TRAIN, '[EPOCH {}] step {}, index {}, epoch_elapsed {}'.format(epoch, step, index, epoch_elapsed))

            if self._lr_curve[0][3] != 0:
                if self._lr_curve[0][0] == '*':
                    self._lr_val *= self._lr_curve[0][1]
                elif self._lr_curve[0][0] == '+':
                    self._lr_val += self._lr_curve[0][1]
                elif self._lr_curve[0][0] == '-':
                    self._lr_val -= self._lr_curve[0][1]
                elif self._lr_curve[0][0] == '/':
                    self._lr_val /= self._lr_curve[0][1]
                elif self._lr_curve[0][0] == '=':
                    self._lr_val = self._lr_curve[0][1]
                else:
                    raise ValueError('The first element of lr curve segment must be (*,+,-,/,=)')

                if self._lr_val < self._lr_minimal:
                    self._lr_val = self._lr_minimal

                dt.train.set_lr_val(self._lr_val)
                dt.train.update_learning_rate(self._optimizer)

                if self._lr_curve[0][3] > 0:
                    self._lr_curve[0][3] -= 1

            if self._lr_curve[0][3] == 0 and len(self._lr_curve) > 1:
                self._lr_curve.pop(0)
                self._epoch_window = self._lr_curve[0][2]

            self._step_elapsed = 0
        return None

    def post_step(self, **kwargs):
        self._step_elapsed += 1
        return None


class TrainCallGroup(dt.util.CallGroup):

    def pre_epoch(self, **kwargs):
        ret = []
        for cb in self._callbacks:
            ret.append(cb.pre_epoch(**kwargs))
        return ret

    def post_epoch(self, **kwargs):
        ret = []
        for cb in self._callbacks:
            ret.append(cb.post_epoch(**kwargs))
        return ret

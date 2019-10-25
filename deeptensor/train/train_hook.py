from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt

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

        self._tqdm = tqdm(total=self._epoch_size, initial=0, desc='train', ncols=80, unit='b', leave=False)
        self._epoch_start = time.time()
        self._num_total = 0

        return None

    def post_step(self, **kwargs):
        size = kwargs['size']
        loss = kwargs['loss']
        acc = kwargs['acc']

        self._num_total += size

        # loss history update
        loss_val = loss.item()
        if not np.isnan(loss_val) and not np.isinf(loss_val):
             if self._ctx.stats.avg_loss is None:
                self._ctx.stats.avg_loss = loss_val
             else:
                self._ctx.stats.avg_loss = self._ctx.stats.avg_loss * 0.9 + loss_val * 0.1

        # acc history update
        if self._ctx.stats.avg_acc is None:
            self._ctx.stats.avg_acc = acc
        else:
            self._ctx.stats.avg_acc = self._ctx.stats.avg_acc * 0.9 + acc * 0.1

        self._tqdm.update(1)

        return None

    def post_epoch(self, **kwargs):
        elapsed_time = time.time() - self._epoch_start

        self._tqdm.close()
        self._tqdm = None

        self._ctx.stats.train_speed = self._num_total / elapsed_time

        return None


class ValidProgressHook(TrainHook):

    def begin(self, **kwargs):
        self._train_start = kwargs['train_start']
        pass

    def pre_epoch(self, **kwargs):
        self._epoch_size = kwargs['epoch_size']

        self._tqdm = tqdm(total=self._epoch_size, initial=0, desc='valid', ncols=80, unit='b', leave=False)
        self._epoch_start = time.time()
        self._num_total = 0
        self._loss_total = 0
        self._acc_total = 0

        return None

    def post_step(self, **kwargs):
        size = kwargs['size']
        loss = kwargs['loss']
        acc = kwargs['acc']

        self._num_total += size
        self._loss_total += loss.item() * size
        self._acc_total += acc * size

        self._tqdm.update(1)

        return None

    def post_epoch(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']

        now_time = time.time()

        self._tqdm.close()
        self._tqdm = None

        self._ctx.stats.valid_speed = self._num_total / (now_time - self._epoch_start)

        if dt.train.is_chief():
            dt.info(dt.DC.TRAIN, '%s Epoch[%03d:lr=%.6f:gs=%06d] train (loss %s, acc %s), valid (loss %s, acc %s), %.3f img/s' %
                                     (time.strftime("%H:%M:%S", time.gmtime(now_time - self._train_start)),
                                     (epoch+1), dt.train.get_lr_val(), step,
                                     ('NA' if self._ctx.stats.avg_loss is None else '%8.6f' % self._ctx.stats.avg_loss),
                                     ('NA' if self._ctx.stats.avg_acc is None else '%8.6f' % self._ctx.stats.avg_acc),
                                     "{:.6f}".format(self._loss_total/self._num_total),
                                     "{:.6f}".format(self._acc_total/self._num_total),
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
        self._epoch_cnt = 0
        self._epoch_window = self._lr_curve[0][2]
        pass

    def pre_epoch(self, **kwargs):
        if self._epoch_window > 0 and self._epoch_cnt >= self._epoch_window:

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

            self._epoch_cnt = 0

        return None

    def post_epoch(self, **kwargs):
        self._epoch_cnt += 1
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


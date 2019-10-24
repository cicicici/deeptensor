from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt

from tqdm import tqdm
import numpy as np
import time


class TrainHook(dt.util.Callback):

    def __init__(self, ctx, **kwargs):
        super(TrainHook, self).__init__(ctx, **kwargs)
        self._every_n_steps = self._ctx.every_n_steps

    def begin(self, **kwargs):
        pass

    def pre_epoch(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        return None

    def post_epoch(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        return None

    def pre_step(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        batch = kwargs['batch']
        return None

    def post_step(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        batch = kwargs['batch']
        return None

    def end(self, **kwargs):
        pass

class TrainProgressHook(TrainHook):

    def begin(self, **kwargs):
        self._train_start = kwargs['train_start']
        pass

    def pre_epoch(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        self._epoch_size = kwargs['epoch_size']
        self._batch_size = kwargs['batch_size']
        self._tqdm = tqdm(total=self._epoch_size, initial=0, desc='train', ncols=80, unit='b', leave=False)
        self._epoch_start = time.time()
        return None

    def post_epoch(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        self._tqdm.close()
        self._tqdm = None
        self._ctx.stats.train_speed = (self._batch_size * self._epoch_size) / (time.time() - self._epoch_start)
        return None

    def post_step(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        batch = kwargs['batch']
        loss = kwargs['loss']
        correct = kwargs['correct']

        # loss history update
        if loss is not None and \
                not np.isnan(loss) and not np.isinf(loss):
            if self._ctx.stats.avg_loss is None:
                self._ctx.stats.avg_loss = loss
            else:
                self._ctx.stats.avg_loss = self._ctx.stats.avg_loss * 0.9 + loss * 0.1

        # acc history update
        if correct is not None:
            if self._ctx.stats.avg_acc is None:
                self._ctx.stats.avg_acc = correct/self._batch_size
            else:
                self._ctx.stats.avg_acc = self._ctx.stats.avg_acc * 0.9 + correct/self._batch_size * 0.1

        self._tqdm.update(1)
        return None


class ValidProgressHook(TrainHook):

    def begin(self, **kwargs):
        self._train_start = kwargs['train_start']
        pass

    def pre_epoch(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        self._epoch_size = kwargs['epoch_size']
        self._batch_size = kwargs['batch_size']
        self._tqdm = tqdm(total=self._epoch_size, initial=0, desc='valid', ncols=80, unit='b', leave=False)
        return None

    def post_epoch(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        train_loader = kwargs['train_loader']
        valid_loader = kwargs['valid_loader']
        loss = kwargs['loss']
        correct = kwargs['correct']

        elapsed_time = time.time() - self._train_start

        self._tqdm.close()
        self._tqdm = None

        if dt.train.is_chief():
            dt.info(dt.DC.TRAIN, '%s Epoch[%03d:lr=%.6f:gs=%06d] train (loss %s, acc %s), valid (loss %s, acc %s), %.3f img/s' %
                                     (time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
                                     (epoch+1), dt.train.get_lr_val(), step,
                                     ('NA' if self._ctx.stats.avg_loss is None else '%8.6f' % self._ctx.stats.avg_loss),
                                     ('NA' if self._ctx.stats.avg_acc is None else '%8.6f' % self._ctx.stats.avg_acc),
                                     "{:.6f}".format(loss/len(valid_loader.dataset)),
                                     "{:.6f}".format(correct/len(valid_loader.dataset)),
                                     self._ctx.stats.train_speed))
        return None

    def post_step(self, **kwargs):
        step = kwargs['step']
        epoch = kwargs['epoch']
        batch = kwargs['batch']
        self._tqdm.update(1)
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


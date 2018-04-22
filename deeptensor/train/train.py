from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf
import horovod.tensorflow as hvd

from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.core.framework.summary_pb2 import Summary

import numpy as np
import time
from tqdm import tqdm


_global_step = None
_learning_rate = None
_lr_val = 0.1

def global_step():
    global _global_step
    return _global_step

def init_gs(opt):
    global _global_step
    _global_step = tf.train.get_or_create_global_step()

def set_lr_val(lr):
    global _lr_val
    _lr_val = lr

def get_lr_val():
    global _lr_val
    return _lr_val

def init_lr(opt):
    global _learning_rate

    _learning_rate = tf.placeholder_with_default(tf.constant(opt.lr_initial, tf.float32), [], name='learning_rate')
    set_lr_val(opt.lr_initial)
    dt.info(dt.DC.TRAIN, 'Initialize learning rate: initial {}, minimal {}, curve {}'
                             .format(opt.lr_initial, opt.lr_minimal, opt.lr_curve))

    # add learning rate summary
    opt.lr = _learning_rate #* hvd.size()
    tf.summary.scalar('learning_r', opt.lr)

def optim(loss, **kwargs):
    opt = dt.Opt(kwargs)

    # default training options
    opt += dt.Opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, momentum=0.9, category='')

    dt.debug(dt.DC.TRAIN, "[OPTIM] {}, lr {}, beta1 {}, beta2 {}, momentum {}, category {}, deferred {}"
                                 .format(opt.optim, opt.lr, opt.beta1, opt.beta2, opt.momentum, opt.catetory, opt.deferred))

    # select optimizer
    if opt.optim == 'MaxProp':
        optimizer = dt.optimize.MaxPropOptimizer(learning_rate=opt.lr, beta2=opt.beta2)
    elif opt.optim == 'AdaMax':
        optimizer = dt.optimize.AdaMaxOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2)
    elif opt.optim == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2)
    elif opt.optim == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(opt.lr, decay=opt.beta1, momentum=opt.momentum)
    elif opt.optim == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(opt.lr, opt.momentum, use_nesterov=True)
    else:
        optimizer = tf.train.GradientDescentOptimizer(opt.lr)

    with tf.name_scope('horovod'):
        hvd_optimizer = hvd.DistributedOptimizer(optimizer)

        grad_op = hvd_optimizer.minimize(loss=loss,
                                         global_step=tf.train.get_global_step())

    return grad_op


class _LearningRateHook(tf.train.SessionRunHook):
    def __init__(self,
                 lr,
                 lr_val=0.001,
                 lr_minimal=1e-6,
                 lr_curve=[[1., 10, 0]], # [[op_0, factor_0, epoch_0, updates_0], ... [op_n, factor_n, epoch_n, update_n]]
                 global_step=None,
                 ep_size=None):
        self._lr = lr
        self._lr_val = lr_val
        self._lr_minimal = lr_minimal
        self._lr_curve = lr_curve
        self._global_step = global_step
        self._ep_size = ep_size

    def begin(self):
        self._gs_start = 0
        self._gs_window = self._ep_size * self._lr_curve[0][2]

    def before_run(self, run_context):
        requests = {}
        requests["global_step"] = self._global_step
        new_lr_val = self._lr_val = get_lr_val()
        if new_lr_val < self._lr_minimal:
            new_lr_val = self._lr_minimal
        return tf.train.SessionRunArgs(requests, feed_dict={self._lr: new_lr_val})

    def after_run(self, run_context, run_values):
        gs_now = run_values.results["global_step"]
        if self._gs_window > 0 and (gs_now - self._gs_start) > self._gs_window:
            self._gs_start = gs_now
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
                    raise ValueError('The first element of lr curve segment must be (*,+,-,/=)')

                set_lr_val(self._lr_val)

                if self._lr_curve[0][3] > 0:
                    self._lr_curve[0][3] -= 1
            if self._lr_curve[0][3] == 0 and len(self._lr_curve) > 1:
                self._lr_curve.pop(0)
                self._gs_window = self._ep_size * self._lr_curve[0][2]


class _TimedSummaryHook(tf.train.SessionRunHook):
    def __init__(self,
                 name="_timed",
                 every_n_steps=100,
                 every_n_secs=None,
                 summary_writer=None,
                 global_step=None,
                 batch_size=None):

        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps'
                             ' and every_n_secs should be provided.')
        self._name = name
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                                every_secs=every_n_secs)
        self._summary_writer = summary_writer
        self._global_step = global_step
        self._batch_size = batch_size

    def begin(self):
        if self._global_step is None:
            raise RuntimeError("Global step should be valid")
        self._step_tag = self._name + "/global_step/sec"
        self._img_tag = self._name + "/img/sec"

    def before_run(self, run_context):
        requests = {"global_step": self._global_step}
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        stale_step = run_values.results["global_step"]
        if self._timer.should_trigger_for_step(stale_step+1):
            cur_step = run_context.session.run(self._global_step)
            if self._timer.should_trigger_for_step(cur_step):
                elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(cur_step)
                if elapsed_time is not None and elapsed_time > 0:
                    steps_per_sec = elapsed_steps / elapsed_time
                    if self._summary_writer is not None:
                        step_summary = Summary(value=[Summary.Value(tag=self._step_tag,
                                                                    simple_value=steps_per_sec)])
                        self._summary_writer.add_summary(step_summary, cur_step)

                        if self._batch_size is not None:
                            imgs_per_sec = self._batch_size * steps_per_sec
                            img_summary = Summary(value=[Summary.Value(tag=self._img_tag,
                                                                       simple_value=imgs_per_sec)])
                            self._summary_writer.add_summary(img_summary, cur_step)


class _ScheduledTaskHook(tf.train.SessionRunHook):
    def __init__(self,
                 name="_scheduled",
                 every_n_steps=None,
                 every_n_secs=None,
                 global_step=None,
                 summary_writer=None,
                 opt=None,
                 task_fn=None,
                 task_ops=None):

        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps'
                             ' and every_n_secs should be provided.')
        self._name = name
        self._timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=every_n_steps,
                                                                every_secs=every_n_secs)
        self._global_step = global_step
        self._summary_writer = summary_writer
        self._opt = opt
        self._task_fn = task_fn
        self._task_ops = task_ops

    def after_create_session(self, session, coord):
        cur_step = session.run(self._global_step)
        cur_ep = int(cur_step // self._opt.data.ep_size)
        cur_ep_step = int(cur_step % self._opt.data.ep_size)
        self._timer.update_last_triggered_step(cur_ep * self._opt.data.ep_size)

    def begin(self):
        if self._global_step is None:
            raise RuntimeError("Global step should be valid")

    def before_run(self, run_context):
        requests = {"global_step": self._global_step}
        if self._task_ops is not None:
            requests = {**requests, **self._task_ops}
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        cur_step = run_values.results["global_step"]
        if self._timer.should_trigger_for_step(cur_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(cur_step)
            if self._task_fn is not None:
                self._task_fn(run_context, run_values, self._opt, elapsed_time, elapsed_steps)

    def end(self, session):
        pass


def init_summary(opt):
    # summary writer
    opt.log_dir = opt.model_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
    opt.summary_writer = tf.summary.FileWriter(opt.log_dir)

def build_train_hooks(opt):
    learning_rate_hook = _LearningRateHook(opt.lr,
                                           lr_val=get_lr_val(),
                                           lr_minimal=opt.lr_minimal,
                                           lr_curve=opt.lr_curve,
                                           global_step=dt.train.global_step(),
                                           ep_size=opt.data.ep_size)

    summary_hook = tf.train.SummarySaverHook(
                        save_steps=opt.summary_steps,
                        save_secs=None,
                        output_dir=opt.log_dir,
                        summary_writer=opt.summary_writer,
                        summary_op=tf.summary.merge_all())

    #logging_hook = tf.train.LoggingTensorHook(
    #                    tensors={'step': model.global_step,
    #                             'loss': model.cost,
    #                             'precision': precision},
    #                    every_n_iter=100)

    timed_summary_hook = _TimedSummaryHook(every_n_steps=opt.summary_steps,
                                           every_n_secs=None,
                                           summary_writer=opt.summary_writer,
                                           global_step=dt.train.global_step(),
                                           batch_size=opt.batch_size)


    def batch_log(context_, values_, opt_, time_, steps_):
        cur_step = int(values_.results["global_step"])
        cur_lr = values_.results["lr"]
        cur_loss = values_.results["loss"]
        cur_acc = values_.results["acc"]
        cur_ep = int(cur_step // opt_.data.ep_size)
        cur_ep_step = int(cur_step % opt_.data.ep_size)

        # loss history update
        if cur_loss is not None and \
                not np.isnan(cur_loss.all()) and not np.isinf(cur_loss.all()):
            if opt_.stats.avg_loss is None:
                opt_.stats.avg_loss = np.mean(cur_loss)
            else:
                opt_.stats.avg_loss = opt_.stats.avg_loss * 0.95 + np.mean(cur_loss) * 0.05

        # acc history update
        if cur_acc is not None:
            if opt_.stats.avg_acc is None:
                opt_.stats.avg_acc = np.mean(cur_acc)
            else:
                opt_.stats.avg_acc = opt_.stats.avg_acc * 0.95 + np.mean(cur_acc) * 0.05

        if dt.util.datalink():
            dt.util.datalink_send_opt(
                    dt.Opt(t='tb',
                           s=cur_step,
                           ep=cur_ep,
                           cs=cur_ep_step,
                           lr=round(float(cur_lr), dt.precision),
                           loss=round(float(cur_loss), dt.precision),
                           acc=round(float(np.mean(cur_acc)), dt.precision),
                           ts=dt.util.get_ts()))

        if opt_.tqdm is None:
            opt_.tqdm = tqdm(total=opt_.data.ep_size, initial=cur_ep_step, desc='train', ncols=80, unit='b', leave=False)
        if cur_ep_step == opt_.data.ep_size - 1:
            opt_.tqdm.close()
            opt_.tqdm = None
        else:
            opt_.tqdm.update(1)


    batch_log_task_hook = _ScheduledTaskHook(every_n_steps=1,
                                             every_n_secs=None,
                                             global_step=dt.train.global_step(),
                                             summary_writer=opt.summary_writer,
                                             opt=opt,
                                             task_fn=batch_log,
                                             task_ops={'lr': opt.lr, 'loss': opt.loss_0, 'acc': opt.acc_0})

    train_hooks = []

    last_step = opt.max_ep * opt.data.ep_size
    train_hooks.append(tf.train.StopAtStepHook(num_steps=None, last_step=last_step))

    #train_hooks.append(tf.train.NanTensorHook(loss))
    train_hooks.append(learning_rate_hook)
    train_hooks.append(summary_hook)
    train_hooks.append(timed_summary_hook)
    train_hooks.append(batch_log_task_hook)

    return train_hooks

def build_chief_train_hooks(opt):
    # metric logging function
    def metric_log(context_, values_, opt_, time_, steps_):
        cur_step = values_.results["global_step"]
        cur_lr = values_.results["lr"]
        cur_time = time.time()
        cur_ep = int(cur_step // opt_.data.ep_size)

        metric_info = ''
        if opt.validate_ep > 0 and cur_ep % opt.validate_ep == 0:
            for i, m in enumerate(opt_.valid_metric):
                if m.cnt <= 0:
                    continue
                m_vals = [0 for op in m.ops]
                m_names = [dt.tensor_short_name(op) for op in m.ops]

                for j in range(0, m.cnt):
                    vals = context_.session.run(m.ops)
                    vals = [round(float(v), dt.precision) for v in vals]
                    for k in range(0, len(m.ops)):
                        m_vals[k] += vals[k]
                    if dt.util.datalink():
                        dt.util.datalink_send_opt(
                                dt.Opt(t='tm',
                                       ep=cur_ep,
                                       name=m.name,
                                       idx=int(j),
                                       vals=vals,
                                       ts=dt.util.get_ts()))
                for k in range(0, len(m.ops)):
                    m_vals[k] = m_vals[k] / m.cnt
                    metric_info += " {}/{} {:.6f},".format(m.name, m_names[k], m_vals[k])
                    m_summary = Summary(value=[Summary.Value(tag="metrics/{}/{}".format(m.name, m_names[k]),
                                                             simple_value=m_vals[k])])
                    opt_.summary_writer.add_summary(m_summary, cur_step)

        dt.info(dt.DC.TRAIN, '\t%s Epoch[%03d:lr=%.6f:gs=%d] loss %s, acc %s,%s %.3f img/s' %
                (time.strftime("%H:%M:%S", time.gmtime(cur_time - opt.train_start)),
                 cur_ep, cur_lr, cur_step,
                 ('NA' if opt_.stats.avg_loss is None else '%8.6f' % opt_.stats.avg_loss),
                 ('NA' if opt_.stats.avg_acc is None else '%8.6f' % opt_.stats.avg_acc),
                 metric_info,
                 float(opt_.batch_size * steps_) / (time_)))

    metric_log_task_hook = _ScheduledTaskHook(every_n_steps=opt.data.ep_size,
                                              every_n_secs=None,
                                              global_step=dt.train.global_step(),
                                              summary_writer=opt.summary_writer,
                                              opt=opt,
                                              task_fn=metric_log,
                                              task_ops={'lr': opt.lr})

    train_hooks = []
    train_hooks.append(metric_log_task_hook)
    return train_hooks

def train(**kwargs):

    opt = dt.Opt(kwargs) + dt.get_ctx()

    # set default train mode
    opt += dt.Opt(is_training=True, is_eval=False, is_pred=False)

    # learning rate
    opt += dt.Opt(lr_initial=0.001, lr_minimal=1e-6, lr_curve=[[0.1, 10, 1]])

    # default training options
    opt += dt.Opt(optim='MaxProp', beta1=0.9, beta2=0.99, momentum=0.9, category='',
                  model_dir='asset/train', tf_random_seed=12345, op_random_seed=12345,
                  max_ep=100000, summary_freq=16, summary_steps=100,
                  save_interval=600, max_keep=5, keep_interval=1000,
                  valid_metric=[], validate_ep=0,
                  tqdm=None)

    # stats
    opt += dt.Opt(stats=dt.Opt(avg_loss=None, avg_acc=None))

    dt.info(dt.DC.TRAIN, '[TRAIN] opt [{}]'
                             .format(opt))

    dt.info(dt.DC.TRAIN, '[HOROVOD] rank {}/{}, local {}'
                             .format(hvd.rank(), hvd.size(), hvd.local_rank()))

    est = opt.est_class(opt, opt.est_cfg)
    tf_est = est.build_estimator(True)

    train_input_fn = est.build_input_fn(True)
    if opt.summary_freq > 0:
        opt.summary_steps = opt.data.ep_size // opt.summary_freq

    opt.train_start = time.time()
    est.pre_train()

    with tf.name_scope('bcast'):
        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    tf_est.train(input_fn=train_input_fn,
                 hooks=[bcast_hook])

    est.post_train()

def restore(sess, save_path, category=''):
    # to list
    if not isinstance(category, (tuple, list)):
        category = [category]

    # make variable list to load
    var_list = {}
    for cat in category:
        for t in tf.global_variables():
            if t.name.startswith(cat):
                var_list[t.name[:-2]] = t

    # restore parameters
    saver = tf.train.Saver(var_list)
    saver.restore(sess, save_path)

def regularizer_loss(scale=1.0):
    losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = scale * tf.reduce_sum(losses)
    #reg_loss = scale * tf.reduce_sum([tf.reduce_mean(loss) for loss in losses])
    return reg_loss


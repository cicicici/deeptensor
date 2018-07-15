from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod

import deeptensor as dt
import tensorflow as tf
import horovod.tensorflow as hvd


class BaseEstimator(object):
    __metaclass__ = ABCMeta

    def __init__(self, opt, cfg):
        self._opt = opt
        self._cfg = cfg

    @abstractmethod
    def build_data(self, is_training):
        return None

    @abstractmethod
    def build_input_fn(self, is_training):
        return None

    @abstractmethod
    def forward(self, tensor, is_training, reuse=False):
        return None

    @abstractmethod
    def define_loss(self, logits, labels, is_training):
        return None

    @abstractmethod
    def define_validation(self):
        return None

    @abstractmethod
    def define_eval_metric_ops(self):
        return None

    @abstractmethod
    def build_train_hooks(self, is_chief):
        return []

    @abstractmethod
    def pre_train(self):
        return None

    @abstractmethod
    def post_train(self):
        return None

    def preprocess_data(self, tensor, is_training):
        return None

    def get_train_op(self, loss):
        grad_op = dt.train.optim(loss, optim=self._opt.optim, lr=self._opt.lr,
                                 beta1=self._opt.beta1, beta2=self._opt.beta2, momentum=self._opt.momentum,
                                 category=self._opt.category)

        update_op = [t for t in tf.get_collection(tf.GraphKeys.UPDATE_OPS)]

        train_op = tf.group(*([grad_op] + update_op))

        return train_op

    def get_model_fn(self):

        def model_fn(features, labels, mode, params):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            is_eval = (mode == tf.estimator.ModeKeys.EVAL)
            is_predict = (mode == tf.estimator.ModeKeys.PREDICT)

            dt.train.init_gs(self._opt)
            dt.train.init_lr(self._opt)
            dt.train.init_summary(self._opt)

            local_init_op = None

            images = features['images']

            with tf.name_scope('logits'):
                with dt.ctx(is_training=is_training):
                    logits = self.forward(images, is_training, reuse=False)

            with tf.name_scope('acc'):
                prob = dt.activation.softmax(logits)
                acc = dt.metric.accuracy(prob, target=labels, name='acc')
                dt.summary_metric(acc)

                if isinstance(acc, (tuple, list)):
                    self._opt.acc_0 = acc[0]
                else:
                    self._opt.acc_0 = acc

                cls = tf.argmax(input=logits, axis=1)

            total_loss = None
            if is_training or is_eval:
                with tf.name_scope('loss'):
                    loss = self.define_loss(logits, labels, is_training)

                    loss_reg = dt.train.regularizer_loss(scale=1.)
                    dt.summary_loss(tf.identity(loss_reg, 'loss_reg'))

                    tf.losses.add_loss(loss)
                    total_loss = tf.losses.get_total_loss()

                    if isinstance(total_loss, (tuple, list)):
                        self._opt.loss_0 = total_loss[0]
                    else:
                        self._opt.loss_0 = total_loss

            train_op = None
            if is_training:
                train_op = self.get_train_op(total_loss)

            # validation
            if is_training:
                with dt.ctx(is_training=False, reuse=True, summary=False):
                    self._opt.valid_metric = self.define_validation()

            predictions_dict = None
            if is_predict:
                predictions_dict = {'logits': logits,
                                    'cls': cls,
                                    'prob': prob}
                for k, v in features.iteritems():
                    predictions_dict[k] = v

            eval_metric_ops = None
            if is_eval:
                with tf.name_scope('eval'):
                    eval_metric_ops = self.define_eval_metric_ops()

            # generate hooks at last
            training_hooks = None
            if is_training:
                training_hooks = dt.train.build_train_hooks(self._opt)
                training_hooks.extend(self.build_train_hooks(False))

                training_chief_hooks = []
                if hvd.rank() == 0:
                    training_chief_hooks = dt.train.build_chief_train_hooks(self._opt)
                    training_chief_hooks.extend(self.build_train_hooks(True))

            saver = tf.train.Saver(max_to_keep=self._opt.max_keep)
            scaffold = tf.train.Scaffold(init_op=None,
                                         init_feed_dict=None,
                                         init_fn=None,
                                         ready_op=None,
                                         ready_for_local_init_op=None,
                                         local_init_op=local_init_op,
                                         summary_op=None,
                                         saver=saver,
                                         copy_from_scaffold=None)

            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions_dict,
                                              loss=total_loss,
                                              train_op=train_op,
                                              eval_metric_ops=eval_metric_ops,
                                              training_hooks=training_hooks,
                                              training_chief_hooks=training_chief_hooks,
                                              scaffold=scaffold)
        return model_fn

    def build_estimator(self, is_training):
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = False
        session_config.allow_soft_placement = True
        session_config.log_device_placement = False
        session_config.gpu_options.visible_device_list = str(self._opt.args.gpu0 + hvd.local_rank())

        save_checkpoints_secs = None
        if dt.train.is_chief():
            save_checkpoints_secs = self._opt.save_interval

        run_config = tf.estimator.RunConfig().replace(
                        model_dir=self._opt.model_dir,
                        tf_random_seed=self._opt.tf_random_seed,
                        save_summary_steps=0,
                        save_checkpoints_steps=None,
                        save_checkpoints_secs=save_checkpoints_secs,
                        session_config=session_config,
                        keep_checkpoint_max=self._opt.max_keep,
                        keep_checkpoint_every_n_hours=self._opt.keep_interval,
                        log_step_count_steps=self._opt.summary_steps)

        return tf.estimator.Estimator(model_fn=self.get_model_fn(),
                                      config=run_config,
                                      params={})

    def train(self):
        pass

    def evaluate(self):
        pass

    def inference(self, tensor, checkpoint, batch_size=None):
        return None


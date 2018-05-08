from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import tensorflow as tf

from deeptensor.estimator import estimator as estimator


class CifarEstimator(estimator.BaseEstimator):

    def __init__(self, opt, cfg):
        super(CifarEstimator, self).__init__(opt, cfg)
        dt.debug(dt.DC.TRAIN, "[EST] {} initialized".format(type(self).__name__))

    def build_data(self, is_training):
        args = self._opt.args
        with tf.name_scope('dataset'):
            data = dt.data.Cifar10TF(batch_size=args.batch_size, valid_size=args.valid_size,
                                     distorted=True, out_height=32, out_width=32,
                                     data_format=self._opt.data_format).init_data()

            ep_size = data.train.num_batch
            v_ep_size = data.valid.num_batch

        return dt.Opt(data=data, ep_size=ep_size, v_ep_size=v_ep_size)

    def build_input_fn(self, is_training):
        self._opt.data = self.build_data(is_training)

        def input_fn(params):
            with tf.name_scope('dataset'):
                self._opt.data.data.generate()
            self._opt.data.tx, self._opt.data.ty = self._opt.data.data.train.images, self._opt.data.data.train.labels
            self._opt.data.vx, self._opt.data.vy = self._opt.data.data.valid.images, self._opt.data.data.valid.labels
            if is_training:
                features = {'images': self._opt.data.tx}
                labels = self._opt.data.ty
            else:
                features = {'images': self._opt.data.vx}
                labels = self._opt.data.vy
            return (features, labels)
        return input_fn

    def forward(self, tensor, is_training, reuse=False):
        args = self._opt.args

        dt.summary_image(tensor)
        if args.model_name == "fitnet4":
            logits = dt.model.fitnet4(tensor, args.class_num)
        elif args.model_name == "resnet":
            if args.model_type == "v1":
                logits = dt.model.resnet_v1(tensor, args.class_num, block_type=args.block_type, blocks=args.blocks,
                                            shortcut=args.shortcut,
                                            regularizer=args.regularizer, conv_decay=args.conv_decay, fc_decay=args.fc_decay)
            elif args.model_type == "v2":
                logits = dt.model.resnet_v2(tensor, args.class_num, block_type=args.block_type, blocks=args.blocks,
                                            shortcut=args.shortcut,
                                            regularizer=args.regularizer, conv_decay=args.conv_decay, fc_decay=args.fc_decay)
        return logits

    def define_loss(self, logits, labels, is_training):
        loss = dt.loss.ce(logits, target=labels, softmax=False)
        return loss

    def define_validation(self):
        valid_metric = []
        args = self._opt.args
        if args.validate_ep > 0:
            with tf.name_scope('valid'):
                vx = self._opt.data.vx
                vy = self._opt.data.vy
                # May not need this ctx line
                with dt.ctx(is_training=False, reuse=True, summary=False):
                    logits_val = self.forward(vx, False, reuse=True)
                loss_val = dt.loss.ce(logits_val, target=vy, softmax=False)
                soft_val = dt.activation.softmax(logits_val)
                acc1_val = tf.reduce_mean(dt.metric.accuracy(soft_val, target=vy), name='acc1')
                valid_metric.append(dt.Opt(name='valid', ops=[loss_val, acc1_val], cnt=self._opt.data.v_ep_size))
        return valid_metric

    def define_eval_metric_ops(self):
        return None

    def build_train_hooks(self, is_chief):
        return []

    def pre_train(self):
        print(self._opt)
        return None

    def post_train(self):
        return None


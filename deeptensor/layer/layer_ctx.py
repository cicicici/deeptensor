from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import types
from functools import wraps

import deeptensor as dt
import tensorflow as tf


def dec_layer_func(func):

    @wraps(func)
    def wrapper(tensor, **kwargs):

        # kwargs parsing
        opt = dt.Opt(kwargs) + dt.get_ctx()

        # set default mode mode
        opt += dt.Opt(is_training=True, reuse=None)

        # options for resnet
        opt += dt.Opt(layer_first=True, shortcut=None)

        # set default argument
        try:
            shape = tensor.get_shape().as_list()

            # batch normalization off, layer normalization off, dropout off
            opt += dt.Opt(shape=shape, in_dim=shape[-1], dim=shape[-1],
                          bn=False, ln=False, dout=0,
                          regularizer=None, weight_decay=1e-6,
                          summary=True, scale=True,
                          weight_filler="he", bn_gamma=1.0, ln_gamma=1.0)

            if opt.regularizer == 'l1':
                opt.regularizer_func = lambda x: tf.reduce_mean(tf.abs(x))
            elif opt.regularizer == 'l2':
                opt.regularizer_func = lambda x: tf.nn.l2_loss(x) * opt.weight_decay
            elif opt.regularizer == 'l2-b':
                opt.regularizer_func = lambda x: tf.square(tf.reduce_mean(tf.square(x)))
            else:
                opt.regularizer_func = None

            assert not (opt.bn and opt.ln), 'one of batch normalization and layer normalization is available.'

            # disable bias when normalization on
            #opt += dt.Opt(bias=not (opt.bn or opt.ln))
            opt += dt.Opt(bias=True)
        finally:
            pass

        # automatic layer naming
        if opt.name is None:
            # layer function name will be used as layer name
            opt.name = func.__name__

            # find existing layer names
            exist_layers = []
            for t in tf.global_variables():
                scope_name = tf.get_variable_scope().name
                prefix = scope_name + '/' if len(scope_name) > 0 else ''
                i = t.name.rfind(prefix + opt.name)
                if i >= 0:
                    exist_layers.append(t.name[i:].split('/')[-2])
            exist_layers = list(set(exist_layers))

            # layer name numbering
            if len(exist_layers) == 0:
                opt.name += '_1'
            else:
                opt.name += '_%d' % (max([int(n.split('_')[-1]) for n in exist_layers]) + 1)

        dt.debug(dt.DC.NET, "[LAYER] {}, T {}, R {}, shape {}, bn {}, ln {}, scale {}, regularizer {}, weight_decay {}, act {}, dout {}, first {}, shortcut {}, bn_gamma {}, ln_gamma {}"
                                 .format(opt.name, opt.is_training, opt.reuse, opt.shape, opt.bn, opt.ln, opt.scale, opt.regularizer, opt.weight_decay, opt.act, opt.dout, opt.layer_first, (opt.shortcut is not None), opt.bn_gamma, opt.ln_gamma))

        with tf.variable_scope(opt.name, reuse=opt.reuse) as scope:

            if opt.layer_first:
                # call layer function
                out = func(tensor, opt)
            else:
                out = tensor
            out_shape = out.get_shape()
            out_dim = dt.utils.get_dim(out)

            # apply batch normalization
            if opt.bn:
                beta = dt.initializer.constant('beta', out_dim, summary=opt.summary)
                gamma = dt.initializer.constant('gamma', out_dim, value=opt.bn_gamma, summary=opt.summary, trainable=opt.scale)

                # offset, scale parameter for inference
                mean_running = dt.initializer.constant('mean_run', out_dim, trainable=False, summary=opt.summary)
                variance_running = dt.initializer.constant('variance_run', out_dim, value=1, trainable=False, summary=opt.summary)

                # use fused batch norm if ndims in [2, 3, 4]
                if out_shape.ndims in [2, 3, 4]:
                    # add HW dims if necessary, fused_batch_norm requires shape to be NHWC
                    if out_shape.ndims == 2:
                        out = tf.expand_dims(out, axis=1)
                        out = tf.expand_dims(out, axis=2)
                    elif out_shape.ndims == 3:
                        out = tf.expand_dims(out, axis=2)

                    fused_eps = dt.eps if dt.eps > 1e-5 else 1e-5
                    if opt.is_training:
                        out, mean, variance = tf.nn.fused_batch_norm(out, gamma, beta, epsilon=fused_eps)
                    else:
                        out, mean, variance = tf.nn.fused_batch_norm(out, gamma, beta, mean=mean_running, variance=variance_running, epsilon=fused_eps, is_training=False)

                    # restore original shape if HW dims was added
                    if out_shape.ndims == 2:
                        out = tf.squeeze(out, axis=[1, 2])
                    elif out_shape.ndims == 3:
                        out = tf.squeeze(out, axis=2)
                # fallback to naive batch norm
                else:
                    mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))
                    if opt.is_training:
                        out = tf.nn.batch_normalization(out, mean, variance, beta, gamma, dt.eps)
                    else:
                        out = tf.nn.batch_normalization(out, mean_running, variance_running, beta, gamma, dt.eps)

                if opt.is_training:
                    # add running mean, variance to UPDATE_OP collection
                    decay = 0.9
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_running.assign(mean_running * decay + mean * (1 - decay)))
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_running.assign(variance_running * decay + variance * (1 - decay)))

            # apply layer normalization
            if opt.ln:
                # offset, scale parameter
                beta = dt.initializer.constant('beta', out_dim, summary=opt.summary)
                if opt.scale:
                    gamma = dt.initializer.constant('gamma', out_dim, value=opt.ln_gamma, summary=opt.summary)

                # calc layer mean, variance for final axis
                mean, variance = tf.nn.moments(out, axes=[len(out.get_shape()) - 1], keep_dims=True)

                # apply normalization
                out = (out - mean) / tf.sqrt(variance + dt.eps)
                # apply parameter
                if opt.scale:
                    out = gamma * out + beta
                else:
                    out = out + beta

            if opt.layer_first and (opt.shortcut is not None):
                dt.debug(dt.DC.NET, "[LAYER] add shortcut [{}], first {}".format(dt.tensor_name(opt.shortcut), opt.layer_first))
                out = out + opt.shortcut

            # apply activation
            if opt.act:
                out = dt.activation.perform(opt.act.lower(), out)

            # apply dropout
            if opt.is_training and opt.dout and (opt.dout > 0 and opt.dout < 1):
                out = tf.nn.dropout(out, 1 - opt.dout),

            if not opt.layer_first:
                # call layer function
                out = func(out, opt)

                if opt.shortcut is not None:
                    dt.debug(dt.DC.NET, "[LAYER] add shortcut [{}], first {}".format(dt.tensor_name(opt.shortcut), opt.layer_first))
                    out = out + opt.shortcut

            # rename tensor
            out = tf.identity(out, 'out')

            # add final output summary
            if opt.summary:
                dt.summary_activation(out)

            # save node info for reuse
            out._sugar = dt.Opt(func=func, arg=dt.Opt(kwargs) + dt.get_ctx(),
                                prev=tensor, is_layer=True, name=opt.name)
            # inject reuse function
            out._reuse = types.MethodType(dt.dt_reuse, out)

        return out

    return wrapper


#! /usr/bin/python
# -*- coding: utf8 -*-

import deeptensor as dt
import tensorflow as tf
import horovod.tensorflow as hvd

# Init horovod
hvd.init()

# Configuration
cfg = dt.util.Config(name="ImageNet")
cfg.dump_config()
ARGS = cfg.opt().args

# Datalink over network
def datalink_recv(socket, packet):
    opt = dt.Opt().loads(packet._data.decode())
    #print(opt)
    # Set learning rate
    if opt.t == 'cmd':
        if opt.a == 'set':
            if opt.key == 'lr':
                dt.train.set_lr_val(opt.val)

    dt.util.datalink().send_opt_sock(socket, dt.Opt(ACK='OK'))

if ARGS.port > 1000 and hvd.rank() == 0:
    dt.util.datalink_start(port=ARGS.port)
    dt.util.datalink_register_recv(datalink_recv)

# Train
with dt.ctx(optim=ARGS.optim, lr_initial=ARGS.lr_initial, lr_minimal=ARGS.lr_minimal,
            lr_curve=ARGS.lr_curve):
    dt.train.train(args=ARGS, est_class = dt.estimator.ImageNetEstimator, est_cfg=dt.Opt(),
                   batch_size=ARGS.batch_size, summary_freq=ARGS.summary_freq,
                   validate_ep=ARGS.validate_ep, max_ep=ARGS.max_ep,
                   model_dir=ARGS.model_dir, save_interval=ARGS.save_interval,
                   tf_random_seed=1234 * (hvd.rank()+1))

dt.util.datalink_close()


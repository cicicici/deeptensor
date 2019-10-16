from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import horovod.torch as hvd

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
    _global_step = 0

def set_lr_val(lr):
    global _lr_val
    _lr_val = lr

def get_lr_val():
    global _lr_val
    return _lr_val

def init_lr(opt):
    global _learning_rate

    #_learning_rate = tf.placeholder_with_default(tf.constant(opt.lr_initial, tf.float32), [], name='learning_rate')
    set_lr_val(opt.lr_initial)
    dt.info(dt.DC.TRAIN, 'Initialize learning rate: initial {}, minimal {}, curve {}'
                             .format(opt.lr_initial, opt.lr_minimal, opt.lr_curve))

    # add learning rate summary
    #opt.lr = _learning_rate #* hvd.size()
    #tf.summary.scalar('learning_r', opt.lr)

def is_chief():
    return hvd.rank() == 0

def init_summary(opt):
    # summary writer
    opt.log_dir = opt.inst_dir + '/run-%02d%02d-%02d%02d' % tuple(time.localtime(time.time()))[1:5]
    #opt.summary_writer = tf.summary.FileWriter(opt.log_dir)

def _close_tqdm(opt):
    if opt.tqdm is not None:
        opt.tqdm.close()
        opt.tqdm = None

def optim_func(loss, **kwargs):
    opt = dt.Opt(kwargs)

    # default training options
    opt += dt.Opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, momentum=0.9, category='')

    dt.debug(dt.DC.TRAIN, "[OPTIM] {}, lr {}, beta1 {}, beta2 {}, momentum {}, category {}, deferred {}"
                                 .format(opt.optim, opt.lr, opt.beta1, opt.beta2, opt.momentum, opt.catetory, opt.deferred))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(**kwargs):

    opt = dt.Opt(kwargs) + dt.get_ctx()

    # set default train mode
    opt += dt.Opt(is_training=True, is_eval=False, is_pred=False)

    # learning rate
    opt += dt.Opt(lr_initial=0.001, lr_minimal=1e-6, lr_curve=[[0.1, 10, 1]])

    # default training options
    opt += dt.Opt(optim='MaxProp', beta1=0.9, beta2=0.99, momentum=0.9, category='',
                  model_dir='asset/train', random_seed=12345, op_random_seed=12345,
                  max_ep=100000, summary_freq=16, summary_steps=100,
                  save_interval=600, max_keep=5, keep_interval=1000,
                  valid_metric=[], validate_ep=0, data_format=dt.dformat.DEFAULT,
                  tqdm=None)

    # stats
    opt += dt.Opt(stats=dt.Opt(avg_loss=None, avg_acc=None))

    dt.info(dt.DC.TRAIN, '[TRAIN] opt')
    dt.print_pp(dt.opt_to_dict(opt))

    dt.info(dt.DC.TRAIN, '[HOROVOD] rank {}/{}, local {}'
                             .format(hvd.rank(), hvd.size(), hvd.local_rank()))

    #if opt.summary_freq > 0:
    #    opt.summary_steps = opt.data.ep_size // opt.summary_freq

    torch.manual_seed(opt.random_seed)

    est = opt.est_class(opt, opt.est_cfg)
    est.build_estimator()

    device = est.device
    #train_loader = est.data.train.loader
    #valid_loader = est.data.valid.loader
    model = est.model

    kwargs = {'num_workers': 1, 'pin_memory': True} if est.use_cuda else {}
    print(kwargs)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('_asset/data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=opt.args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST('_asset/data/mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=opt.args.valid_size, shuffle=True, **kwargs)
    #model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=opt.lr_initial, momentum=opt.momentum)

    est.pre_train()
    train_start = time.time()

    #bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    for epoch in range(1, 10 + 1):

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(valid_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))

    train_end = time.time()
    est.post_train()
    print(time.strftime("%H:%M:%S", time.gmtime(train_end - train_start)))

    #if (args.save_model):
    #    torch.save(est.model().state_dict(), "mnist_cnn.pt")


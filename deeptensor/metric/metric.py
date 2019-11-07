from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch


def accuracy(logits, labels, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def save(optimizer, fname, **kwargs):
    params = dt.Opt(kwargs)

    torch.save({'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_params': params.to_dict(),
               }, fname)

def load(optimizer, fname):
    if dt.util.file_exist(fname):
        checkpoint = torch.load(fname)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return dt.Opt.from_dict(checkpoint['optimizer_params'])
    else:
        return None


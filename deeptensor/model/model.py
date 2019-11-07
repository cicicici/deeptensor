from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import torch.nn as nn
import torch.nn.functional as F


def save(model, fname, **kwargs):
    params = dt.Opt(kwargs)

    torch.save({'model_state_dict': model.state_dict(),
                'model_params': params.to_dict(),
               }, fname)

def load(model, fname):
    if dt.util.file_exist(fname):
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        return dt.Opt.from_dict(checkpoint['model_params'])
    else:
        return None

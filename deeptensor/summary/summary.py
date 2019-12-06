'''
Credits:
https://github.com/sksq96/pytorch-summary
https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt

import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

from torch.nn.modules.module import _addindent


def patch_add_dt(module, gc):
    class_name = module.__class__.__name__
    #dt.trace(dt.DC.TRAIN, "[PATCH] level {}, path {}, key {}, class {}".format(
    #                      gc.level, gc.path, gc.key, class_name))
    if not hasattr(module, '_dt_'):
        module._dt_ = dt.Opt()
    module._dt_.level = gc.level
    module._dt_.path = gc.path
    module._dt_.key = gc.key
    module._dt_.class_name = class_name

def patch_clear_dt(module, gc):
    if hasattr(module, '_dt_'):
        delattr(module, '_dt_')

def _walk_module(cl, module):
    gc = dt.get_ctx_cl(cl)

    if gc.patch_fn:
        gc.patch_fn(module, gc)

    for idx, (key, mod) in enumerate(module._modules.items()):
        if gc.key is None:
            path = gc.path
        else:
            if len(gc.path) > 0:
                path = gc.path + '/' + gc.key
            else:
                path = gc.key
        with dt.ctx_cl(cl, None, level=gc.level+1, key=key, idx=idx, path=path):
            _walk_module(cl, mod)

def summary_model_patch(model, patch_fn=patch_add_dt, **kwargs):
    class_name = model.__class__.__name__

    state = dt.Opt()
    cl = dt.create_ctx_list(args=dt.Opt(kwargs), patch_fn=patch_fn, state=state)

    with dt.ctx_cl(cl, None, level=0, key=None, path=''):
        _walk_module(cl, model)

    return True

def summary_model_fwd(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            if hasattr(module, '_dt_'):
                m_key = "{}:{}/{}".format(module._dt_.level, module._dt_.path, module._dt_.key)
            else:
                m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            if hasattr(module, '_dt_'):
                summary[m_key]["class_name"] = module._dt_.class_name
            else:
                summary[m_key]["class_name"] = class_name

            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            summary[m_key]["weight_size"] = None
            summary[m_key]["bias_size"] = None
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]["weight_size"] = list(module.weight.size())
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["bias_size"] = list(module.bias.size())
            summary[m_key]["nb_params"] = params

        if not (module == model):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    tmpstr =  "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    tmpstr += '\n'
    tmpstr+= "{:<50}  Input: {}".format(model.__class__.__name__, input_size)
    tmpstr += '\n'
    tmpstr += "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    tmpstr += '\n'
    line_new = "{:<50}  {:<30} {:>20} {:>20} {:>20} {:>10} {:>15}".format(
               "Layer Name", "Layer Type", "Input Shape", "Output Shape", "Weight Size", "Bias Size", "Param #")
    tmpstr += line_new
    tmpstr += '\n'
    tmpstr += "============================================================================================================================================================================"
    tmpstr += '\n'
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:<50}  {:30} {:>20} {:>20} {:>20} {:>10} {:>15}".format(
            layer,
            summary[layer]["class_name"],
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            (str(summary[layer]["weight_size"] if summary[layer]["weight_size"] is not None else "")),
            (str(summary[layer]["bias_size"]) if summary[layer]["bias_size"] is not None else ""),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        tmpstr += line_new
        tmpstr += '\n'

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    tmpstr += "============================================================================================================================================================================"
    tmpstr += '\n'
    tmpstr += "Total params: {0:,}".format(total_params)
    tmpstr += '\n'
    tmpstr += "Trainable params: {0:,}".format(trainable_params)
    tmpstr += '\n'
    tmpstr += "Non-trainable params: {0:,}".format(total_params - trainable_params)
    tmpstr += '\n'
    tmpstr += "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    tmpstr += '\n'
    tmpstr += "Input size (MB): %0.2f" % total_input_size
    tmpstr += '\n'
    tmpstr += "Forward/backward pass size (MB): %0.2f" % total_output_size
    tmpstr += '\n'
    tmpstr += "Params size (MB): %0.2f" % total_params_size
    tmpstr += '\n'
    tmpstr += "Estimated Total Size (MB): %0.2f" % total_size
    tmpstr += '\n'
    tmpstr += "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    tmpstr += '\n'

    # return summary
    return tmpstr

def summary_model_wass(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'

    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [torch.nn.modules.container.Container,
                            torch.nn.modules.container.Sequential]:
            modstr = summary_model_wass(module, show_weights=show_weights, show_parameters=show_parameters)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

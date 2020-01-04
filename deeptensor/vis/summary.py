from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deeptensor as dt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


_summary_writer = None

def set_default_writer(writer):
    global _summary_writer
    _summary_writer = writer

def get_default_writer():
    global _summary_writer
    return _summary_writer

def add_graph(model, images, writer=get_default_writer()):
    if writer is None:
        writer = get_default_writer()
    writer.add_graph(model, images)

def add_image(tag, image, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_image(tag, image, global_step=global_step)

def add_images_grid(tag, images, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    grid = torchvision.utils.make_grid(images)
    writer.add_image(tag, grid, global_step=global_step)

def add_images(tag, images, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_images(tag, images, global_step=global_step)

def add_scalar(tag, scalar_value, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_scalar(tag, scalar_value, global_step=global_step)

def add_histogram(tag, values, bins='tensorflow', global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_histogram(tag, values, global_step=global_step, bins=bins)

def add_figure(tag, figure, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_figure(tag, figure, global_step=global_step)

def add_video(tag, video_tensor, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_video(tag, video_tensor, global_step=global_step)

def add_audio(tag, snd_tensor, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_audio(tag, snd_tensor, global_step=global_step)

def add_text(tag, text_string, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_text(tag, text_string, global_step=global_step)

def summary_tensor(tag, tensor, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    writer.add_scalar(tag, torch.mean(tensor).item(), global_step=global_step)
    writer.add_histogram(tag+'-h', tensor, global_step=global_step)

def summary_tensor_abs(tag, tensor, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    tensor_abs = torch.abs(tensor)
    writer.add_scalar(tag, torch.mean(tensor_abs).item(), global_step=global_step)
    writer.add_histogram(tag+'-h', tensor_abs, global_step=global_step)

def summary_tensor_clamp(tag, tensor, min_val, max_val, global_step=None, writer=None):
    if writer is None:
        writer = get_default_writer()
    if global_step is None:
        global_step = dt.train.mono_step()

    tensor_abs = torch.clamp(tensor, min_val, max_val)
    writer.add_scalar(tag, torch.mean(tensor_abs).item(), global_step=global_step)
    writer.add_histogram(tag+'-h', tensor_abs, global_step=global_step)

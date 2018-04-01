#! /usr/bin/python
# -*- coding: utf8 -*-

import sys
import math

from bokeh.models import ColumnDataSource, HoverTool, SaveTool
from bokeh.models.widgets import TextInput, Button, Slider
from bokeh.models import LinearAxis, Range1d
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column, widgetbox

import deeptensor as dt
import tensorflow as tf

# Configuration
cfg = dt.util.Config(name="Sniffer", app="sniffer")
dt.dbg_cfg(level=cfg.opt().debug.level)
cfg.dump_config()
ARGS = cfg.opt().args

# Datalink
opt_buf=[]
opt_buf.append([])
opt_buf.append([])
tb_fill_idx = 0
def datalink_recv(socket, packet):
    opt = dt.Opt().loads(packet._data.decode())
    opt_buf[tb_fill_idx].append(opt)
    print(opt)

dt.util.datalink_start(ARGS.host, ARGS.port)
dt.util.datalink_register_recv(datalink_recv)
#dt.util.datalink().send_opt(dt.Opt(a=0, b='hello'))

# Global
data_tb = None
data_tm = None
g_lr_base = 0.01
g_lr_scale = 0
g_lr = 0

# Data
def init_data():
    global data_tb
    global data_tm

    data_tb = ColumnDataSource(dict(time=[], display_time=[], lr=[], lr_log=[], loss=[], acc=[], ep_step=[]))
    data_tm = ColumnDataSource(dict(time=[], display_time=[], loss_val=[], top1_val=[], top5_val=[], ep_idx=[]))

def clear_data():
    global data_tb
    global data_tm

    data_tb.data = dict(time=[], display_time=[], lr=[], lr_log=[], loss=[], acc=[], ep_step=[])
    data_tm.data = dict(time=[], display_time=[], loss_val=[], top1_val=[], top5_val=[], ep_idx=[])

init_data()

# Chart
def set_lr(lr):
    global g_lr
    if True or g_lr != lr:
        g_lr = lr
        dt.util.datalink().send_opt(dt.Opt(t='cmd', a='set', key='lr', val=g_lr))
        lr_plot.title.text = "Training lr={}".format(g_lr)

def update_lr_base():
    global g_lr_base
    global g_lr_scale

    print("[{}]".format(t_lr_base.value))
    g_lr_base = float(t_lr_base.value)
    s_lr_scale.value = 0

    g_lr_scale = 0
    set_lr(g_lr_base)

def update_lr_scale(attr, old, new):
    global g_lr_base
    global g_lr_scale

    g_lr_scale = new
    lr = g_lr_base * pow(10, g_lr_scale)
    set_lr(lr)

def update_data():
    global opt_buf
    global tb_fill_idx
    buf = opt_buf[tb_fill_idx]
    tb_fill_idx = (tb_fill_idx + 1) % 2

    print("opt_buf, {}, {}".format(len(opt_buf[0]), len(opt_buf[1])))

    new_tb = dt.Opt(time=[], display_time=[], lr=[], lr_log=[], loss=[], acc=[], ep_step=[])
    new_tm = dt.Opt(time=[], display_time=[], loss_val=[], top1_val=[], top5_val=[], ep_idx=[])
    for opt in buf:
        if opt.t == 'tb':
            new_tb.time.append(opt.ts)
            new_tb.display_time.append("{}".format(opt.ts))
            new_tb.lr.append(opt.lr)
            new_tb.lr_log.append(math.log(opt.lr, 10))
            new_tb.loss.append(opt.loss)
            new_tb.acc.append(opt.acc)
            new_tb.ep_step.append("{} ({})".format(opt.ep, opt.s))
        elif opt.t == 'tm':
            new_tm.time.append(opt.ts)
            new_tm.display_time.append("{}".format(opt.ts))
            new_tm.loss_val.append(opt.vals[0])
            new_tm.top1_val.append(opt.vals[1])
            new_tm.top5_val.append(opt.vals[2])
            new_tm.ep_idx.append("{} ({})".format(opt.ep, opt.idx))
    buf.clear()

    data_tb.stream(dict(time=new_tb.time,
                        display_time=new_tb.display_time,
                        lr=new_tb.lr,
                        lr_log=new_tb.lr_log,
                        loss=new_tb.loss,
                        acc=new_tb.acc,
                        ep_step=new_tb.ep_step), 20000)

    data_tm.stream(dict(time=new_tm.time,
                        display_time=new_tm.display_time,
                        loss_val=new_tm.loss_val,
                        top1_val=new_tm.top1_val,
                        top5_val=new_tm.top5_val,
                        ep_idx=new_tm.ep_idx), 10000)

hover_tb = HoverTool(tooltips=[
    ("Time", "@display_time"),
    ("Lr", "@lr"),
    ("Loss", "@loss"),
    ("Acc", "@acc"),
    ("Ep_step", "@ep_step"),
    ("Loss val", "@loss_val"),
    ("Top1", "@top1_val"),
    ("Top5", "@top5_val"),
    ("Ep_idx", "@ep_idx"),
    ])

lr_plot = figure(plot_width=1200,
                 plot_height=300,
                 #x_axis_type='datetime',
                 tools=['pan','wheel_zoom','box_zoom','save','reset', hover_tb],
                 title="Real-Time Plot")

lr_plot.circle(source=data_tb, x='time', y='lr_log', size=3, color="salmon", alpha=0.4)
lr_plot.xaxis.axis_label = "Time"
lr_plot.yaxis.axis_label = "Lr Log10"
lr_plot.title.text = "Training lr"

loss_plot = figure(plot_width=1200,
                   plot_height=400,
                   tools=['pan','wheel_zoom','box_zoom','save','reset', hover_tb],
                   title="Real-Time Plot")

loss_plot.circle(source=data_tb, x='time', y='loss', size=3, color="royalblue", alpha=0.4)
loss_plot.circle(source=data_tm, x='time', y='loss_val', size=3, color="crimson", alpha=0.4)
loss_plot.xaxis.axis_label = "Time"
loss_plot.yaxis.axis_label = "Loss"
loss_plot.title.text = "Training batch loss"

#loss_plot.extra_y_ranges = {"lr_log": Range1d(start=-6, end=1.)}
#loss_plot.circle(source=data_tb, x='time', y='lr_log', size=3, color="red", y_range_name="lr_log", alpha=0.5)
#loss_plot.add_layout(LinearAxis(y_range_name="lr_log"), 'left')

hover_tm = HoverTool(tooltips=[
    ("Time", "@display_time"),
    ("Acc", "@acc"),
    ("Ep_step", "@ep_step"),
    ("Loss val", "@loss_val"),
    ("Top1", "@top1_val"),
    ("Top5", "@top5_val"),
    ("Ep_idx", "@ep_idx"),
    ])

acc_plot = figure(plot_width=1200,
                  plot_height=400,
                  tools=['pan','wheel_zoom','box_zoom','save','reset', hover_tm],
                  title="Real-Time Plot")

acc_plot.circle(source=data_tb, x='time', y='acc', size=3, color="royalblue", alpha=0.4)
acc_plot.circle(source=data_tm, x='time', y='top1_val', size=3, color="crimson", alpha=0.4)
acc_plot.circle(source=data_tm, x='time', y='top5_val', size=3, color="limegreen", alpha=0.4)
acc_plot.xaxis.axis_label = "Time"
acc_plot.yaxis.axis_label = "Acc"
acc_plot.title.text = "Training acc"

# Widgets
t_lr_base = TextInput(placeholder="Base LR")
t_lr_base.title = 'Base LR'
t_lr_base.value = '0.01'
b_update = Button(label="Update")
b_update.on_click(update_lr_base)
s_lr_scale = Slider(start=-4, end=2, value=0, step=.01, title="LR Scale")
s_lr_scale.on_change("value", update_lr_scale)
b_clear_data = Button(label="Clear")
b_clear_data.on_click(clear_data)

inputs_0 = widgetbox([s_lr_scale], width=600)
inputs_1 = widgetbox([t_lr_base], width=200)
inputs_2 = widgetbox([b_update], width=200)
inputs_3 = widgetbox([b_clear_data], width=200)

# Document
curdoc().add_root(column(row(inputs_0, inputs_1, inputs_2, inputs_3), lr_plot, loss_plot, acc_plot, width=1200))
curdoc().title = "Sniffer"
curdoc().add_periodic_callback(update_data, ARGS.update_ms)


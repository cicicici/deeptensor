from .train import init_library, init_device, use_cuda, device, device_index, \
                   device_count, chief_rank, is_chief, is_mp, global_step, global_step_inc, \
                   init_global_step, set_global_step, set_lr_val, get_lr_val, \
                   set_lr_val_mp, get_lr_val_base, mp_average, mp_broadcast, \
                   init_learning_rate, init_summary, init_saver, adjust_learning_rate, \
                   update_learning_rate, dump_learning_rate, train
from .train_hook import TrainHook, TrainProgressHook, ValidProgressHook, LearningRateHook, \
                        TrainCallGroup

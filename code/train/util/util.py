from model.genetic_unet.genetic_unet import Net

import multiprocessing as mp
import multiprocessing.pool
import torch
from deap import tools
import time
from ..train_model import train_one_model

class NoDaemonProcess(mp.Process):

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


import numpy as np
import random


def func_try(population, ind_num, device, model_settings):
    i = 0
    seed = 12
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device=device)
    while True:
        i += 1
        mem_max_cached = torch.cuda.max_memory_cached(device=device) / 1000 ** 3
        mem_used_cached = torch.cuda.memory_cached(device=device) / 1000 ** 3
        torch.cuda.reset_max_memory_cached(device=device)
        torch.cuda.reset_max_memory_allocated(device=device)
        if i > 5:
            break
        if mem_max_cached > 9 and mem_used_cached > 1:
            curr_device = torch.cuda.current_device()
            torch.cuda.empty_cache()
            time.sleep(3)
        else:
            break
    temp = population[ind_num][0:150]
    population[ind_num] = tools.mutFlipBit(population[ind_num], indpb=0.3)[0]
    population[ind_num][0:150] = temp
    model = Net(gene=population[ind_num][:], model_settings=model_settings)
    print('Have changed the channel number!')

    return model, device


def help_func(optimizer_name, learning_rate, l2_weight_decay, gen_num, ind_num, model, batch_size, epochs, device,
              train_set_name, valid_set_name,
              train_set_root, valid_set_root, exp_name,
              population, model_settings):
    metrics, flag = train_one_model(optimizer_name, learning_rate, l2_weight_decay, gen_num, ind_num, model, batch_size,
                                    epochs, device, train_set_name,
                                    valid_set_name,
                                    train_set_root, valid_set_root, exp_name)
    if flag == False:
        while True:
            model, device = func_try(population, ind_num, device, model_settings)
            metrics, flag = train_one_model(optimizer_name, learning_rate, l2_weight_decay, gen_num, ind_num, model,
                                            batch_size, epochs, device,
                                            train_set_name, valid_set_name,
                                            train_set_root, valid_set_root, exp_name)
            if flag == True:
                break

    return metrics


def util_function(i):
    return help_func(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11], i[12], i[13], i[14],
                     i[15])

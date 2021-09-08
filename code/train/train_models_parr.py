import numpy as np
from model.genetic_unet.genetic_unet import Net
import pickle
from .util.util import NoDaemonProcessPool, util_function
import sys
import torch, random
import os
import multiprocessing as mp

sys.path.append('../')

def train_population_parr(train_list, gen_num, population, batch_size, devices, epochs, exp_name, train_set_name,
                          valid_set_name, train_set_root, valid_set_root,
                          optimizer_name, learning_rate, l2_weight_decay, model_settings):
    seed = 12
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_list = []
    metrics_ = []
    pickle_file = open(
        os.path.join(os.path.abspath('.'), 'exps/{}/pickle/gens_{}individuals_code.pkl'.format(exp_name, gen_num)),
        'wb')
    assert len(train_list) == len(population)
    for individual, inds in zip(population, train_list):
        list_ = {'gens_{}_individual_{}'.format(gen_num, inds): individual[:]}
        pickle.dump(list_, pickle_file)
        model_list.append(Net(gene=individual[:], model_settings=model_settings))
    pickle_file.close()
    gpu_num = len(devices)
    for i in np.arange(0, len(population), gpu_num):
        process_num = np.min((i + gpu_num, len(population))) - i
        pool = NoDaemonProcessPool(process_num)
        args = [
            (optimizer_name, learning_rate, l2_weight_decay, gen_num, train_list[i + j], model_list[i + j], batch_size,
             epochs, devices[j],
             train_set_name, valid_set_name,
             train_set_root, valid_set_root, exp_name, population, model_settings)
            for j in
            range(process_num)]
        metrics = pool.map(util_function, args)
        pool.terminate()
        metrics_.extend(metrics)
    return metrics_

import os
import sys
import numpy as np
from scipy.special import comb
import random


def find_train_inds(population):
    i = 0
    train_list = []
    for ind in population:
        if ind.fitness.valid == False:
            train_list.append(i)
        i += 1
    return train_list


def special_initialization(population, code_list):
    for ind, code in zip(population, code_list):
        ind[:] = code
    return population


def save_evolution_stat_ckpt(evolution_stat_dict, exp_name, g):
    import pickle
    pickle_file1 = open(
        os.path.join(os.path.abspath('.'), 'exps/{}/pickle/gens{}_evolution_stat_dict.pkl'.format(exp_name, g)),
        'wb')
    pickle.dump(evolution_stat_dict, pickle_file1)
    pickle_file1.close()


def reload_evolution_stat_ckpt(exp_name, g):
    import pickle
    pickle_file = open(
        os.path.join(os.path.abspath('.'), 'exps/{}/pickle/gens{}_evolution_stat_dict.pkl'.format(exp_name, g)), 'rb')
    pkl2 = pickle.load(pickle_file)
    pickle_file.close()
    evolution_stat_dict = pkl2

    return evolution_stat_dict


def save_population_ckpt(population, exp_name, g):
    import pickle
    pickle_file1 = open(os.path.join(os.path.abspath('.'), 'exps/{}/pickle/gens{}_ckpt.pkl'.format(exp_name, g)),
                        'wb')
    pickle.dump(population, pickle_file1)
    pickle_file1.close()


def reload_population_ckpt(exp_name, g):
    import pickle
    pickle_file = open(os.path.join(os.path.abspath('.'), 'exps/{}/pickle/gens{}_ckpt.pkl'.format(exp_name, g)), 'rb')
    pkl2 = pickle.load(pickle_file)
    pickle_file.close()
    population = pkl2

    return population


def check_dir(exp_name):
    exps_path = os.path.abspath('.')
    ckpt_path = os.path.join(exps_path, 'exps/{}/ckpt'.format(exp_name))
    runs_path = os.path.join(exps_path, 'exps/{}/runs'.format(exp_name))
    pickle_path = os.path.join(exps_path, 'exps/{}/pickle'.format(exp_name))
    csv_path = os.path.join(exps_path, 'exps/{}/csv'.format(exp_name))

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    if not os.path.exists(runs_path):
        os.makedirs(runs_path, exist_ok=True)
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path, exist_ok=True)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path, exist_ok=True)


def get_gene_len(de_func_type, en_func_type, de_node_num_list, en_node_num_list, only_en=False):
    de_func_type_num = len(de_func_type)
    en_func_type_num = len(en_func_type)

    de_node_func_gene_len = int(np.ceil(np.log2(de_func_type_num)))
    en_node_func_gene_len = int(np.ceil(np.log2(en_func_type_num)))

    de_connect_gene_len_list = [None for _ in range(len(de_node_num_list))]
    en_connect_gene_len_list = [None for _ in range(len(en_node_num_list))]

    for i in range(len(de_node_num_list)):
        de_connect_gene_len_list[i] = int(comb(de_node_num_list[i], 2))
    for i in range(len(en_node_num_list)):
        en_connect_gene_len_list[i] = int(comb(en_node_num_list[i], 2))

    de_gene_len_list = [None for _ in range(len(de_node_num_list))]
    en_gene_len_list = [None for _ in range(len(en_node_num_list))]

    for i in range(len(de_node_num_list)):
        de_gene_len_list[i] = de_node_func_gene_len + de_connect_gene_len_list[i]
    for i in range(len(en_node_num_list)):
        en_gene_len_list[i] = en_node_func_gene_len + en_connect_gene_len_list[i]

    if only_en:
        gene_len = sum(en_gene_len_list)
    else:
        gene_len = sum(de_gene_len_list) + sum(en_gene_len_list)

    return gene_len


def bin(n):
    result = ''
    if n:
        result = bin(n // 2)
        return result + str(n % 2)
    else:
        return result


def cxMultiPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoints = []
    for _ in range(10):
        point = random.randint(0, size)
        while point in cxpoints:
            point = random.randint(0, size)
        cxpoints.append(point)
    cxpoints.sort()
    cxpoint1, cxpoint2, cxpoint3, cxpoint4, cxpoint5, cxpoint6, cxpoint7, cxpoint8, cxpoint9, cxpoint10 = cxpoints
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    ind1[cxpoint3:cxpoint4], ind2[cxpoint3:cxpoint4] \
        = ind2[cxpoint3:cxpoint4], ind1[cxpoint3:cxpoint4]
    ind1[cxpoint5:cxpoint6], ind2[cxpoint5:cxpoint6] \
        = ind2[cxpoint5:cxpoint6], ind1[cxpoint5:cxpoint6]
    ind1[cxpoint7:cxpoint8], ind2[cxpoint7:cxpoint8] \
        = ind2[cxpoint7:cxpoint8], ind1[cxpoint7:cxpoint8]
    ind1[cxpoint9:cxpoint10], ind2[cxpoint9:cxpoint10] \
        = ind2[cxpoint9:cxpoint10], ind1[cxpoint9:cxpoint10]

    return ind1, ind2


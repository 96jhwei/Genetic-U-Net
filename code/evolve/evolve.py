import random
from deap import base
from deap import creator
from deap import tools
import torch.multiprocessing
import pickle
import os
from tensorboardX import SummaryWriter
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from train.train_models_parr import train_population_parr
from .util.util import reload_population_ckpt, find_train_inds, check_dir, save_population_ckpt, get_gene_len, cxMultiPoint
import sys
import torch

# sys.path.append('../')

def evolve():
    gpu_num = 4
    seed = 12
    random.seed(seed)
    np.random.seed(seed)
    optimization_objects = ['f1_score']
    optimization_weights = [1]

    channel = 20
    en_node_num = 5
    de_node_num = 5
    sample_num = 4
    exp_name = 'test'
    crossover_rate = 0.9
    mutation_rate = 0.7
    flipping_rate = 0.05
    gens = 60
    epochs = 3
    batch_size = 1
    parents_num = 2
    offsprings_num = 2
    devices = [torch.device(type='cuda', index=i) for i in range(gpu_num)]
    optimizer_name = 'Lookahead(Adam)'
    learning_rate = 0.001
    l2_weight_decay = 0

    resume_train = False
    train_set_name = 'DRIVE'
    valid_set_name = 'DRIVE'
    train_set_root = os.path.join(os.path.abspath('.'), 'dataset', 'trainset', train_set_name)
    valid_set_root = os.path.join(os.path.abspath('.'), 'dataset', 'validset', valid_set_name)

    en_node_num_list = [en_node_num for _ in range(sample_num + 1)]
    de_node_num_list = [de_node_num for _ in range((sample_num))]

    func_type = ['conv_bn_relu_3', 'conv_bn_mish_3', 'conv_in_relu_3',
                 'conv_in_mish_3', 'p_conv_bn_relu_3', 'p_conv_bn_mish_3',
                 'p_conv_in_relu_3', 'p_conv_in_mish_3', 'conv_bn_relu_5',
                 'conv_bn_mish_5', 'conv_in_relu_5','conv_in_mish_5', 'p_conv_bn_relu_5',
                 'p_conv_bn_mish_5','p_conv_in_relu_5', 'p_conv_in_mish_5']

    gene_len = get_gene_len(de_func_type=func_type, en_func_type=func_type, de_node_num_list=de_node_num_list,
                            en_node_num_list=en_node_num_list, only_en=False)

    model_settings = {'channel': channel, 'en_node_num_list': en_node_num_list, 'de_node_num_list': de_node_num_list,
                      'sample_num': sample_num, 'en_func_type': func_type, 'de_func_type': func_type}

    creator.create("FitnessMax", base.Fitness, weights=optimization_weights)
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, gene_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutateL", tools.mutFlipBit, indpb=flipping_rate)

    check_dir(exp_name)
    sum_writer = SummaryWriter(log_dir=os.path.join(os.path.abspath('.'), 'exps/{}/runs'.format(exp_name)))

    if resume_train:
        g = 1
        exp_name_load = None
        population = reload_population_ckpt(exp_name_load, g=g)

        for i in range(len(population)):
            if not os.path.exists(
                    os.path.join(os.path.abspath('.'), 'exps/{}/ckpt/individual_{}'.format(exp_name, i))):
                os.mkdir(os.path.join(os.path.abspath('.'), 'exps/{}/ckpt/individual_{}'.format(exp_name, i)))
        if not os.path.exists(os.path.join(os.path.abspath('.'), 'exps/{}/pickle/'.format(exp_name))):
            os.mkdir(os.path.join(os.path.abspath('.'), 'exps/{}/pickle/'.format(exp_name)))
        offspring = None
        sum_writer = SummaryWriter(log_dir=os.path.join(os.path.abspath('.'), 'exps/{}/runs'.format(exp_name)))

    else:
        population = toolbox.population(n=parents_num)
        print('==========Sucessfully initialize population==========')

        for i in range(len(population)):
            if not os.path.exists(os.path.join(os.path.abspath('.'), 'exps/{}/ckpt/individual_{}'.format(exp_name, i))):
                os.mkdir(os.path.join(os.path.abspath('.'), 'exps/{}/ckpt/individual_{}'.format(exp_name, i)))
        if not os.path.exists(os.path.join(os.path.abspath('.'), 'exps/{}/pickle/'.format(exp_name))):
            os.mkdir(os.path.join(os.path.abspath('.'), 'exps/{}/pickle/'.format(exp_name)))

        train_list = find_train_inds(population)
        print('gens_{} train individuals is:'.format(0), train_list)

        metrics = train_population_parr(train_list=train_list, gen_num=0, population=population, batch_size=batch_size,
                                        devices=devices, epochs=epochs, exp_name=exp_name,
                                        train_set_name=train_set_name,
                                        valid_set_name=valid_set_name, train_set_root=train_set_root,
                                        valid_set_root=valid_set_root, optimizer_name=optimizer_name,
                                        learning_rate=learning_rate,
                                        model_settings=model_settings, l2_weight_decay=l2_weight_decay)

        for i in range(len(population)):
            fitness = []
            for opt_obj in optimization_objects:
                fitness.append(metrics[i][opt_obj])
            population[i].fitness.values = fitness

        print('evaluate gens_{} successfully'.format(0))
        save_population_ckpt(population=population, exp_name=exp_name, g=0)

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        g = 0
        sum_writer.add_scalar('best_fitness', tools.selBest(population,k=1)[0].fitness.values[0], g)
        offspring = None

    for n in range(g + 1, gens):

        from copy import deepcopy
        parents = deepcopy(population)
        new_parents = list(map(toolbox.clone, parents))
        if offspring != None:
            del offspring
        offspring = toolbox.population(n=offsprings_num)
        if len(new_parents) >= 2:
            for i in range(int(np.ceil(offsprings_num // 2))):
                if random.random() < crossover_rate:
                    for _ in range(10):
                        new_parents_list = deepcopy(tools.selTournament(new_parents, 2, tournsize=2))
                        gene_len = len(new_parents_list[0])
                        xor_result = []
                        for p in range(gene_len):
                            xor_result.append(int(new_parents_list[0][p]) ^ int(new_parents_list[1][p]))
                        sim = sum(xor_result) / gene_len
                        if sim > 0.2:
                            break
                    off1, off2 = cxMultiPoint(new_parents_list[0], new_parents_list[1])
                else:
                    new_parents_list = deepcopy(tools.selTournament(new_parents, 2, tournsize=2))
                    off1, off2 = new_parents_list[0], new_parents_list[1]
                    
                offspring[i][:] = off1[:]
                offspring[i + 1][:] = off2[:]
                del off1.fitness.values
                del off2.fitness.values
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

                del new_parents_list
            offspring = offspring[:offsprings_num]

            for i in range(offsprings_num):
                pb = mutation_rate
                if random.random() < pb:
                    offspring[i][:] = toolbox.mutateL(offspring[i])[0]
                    del offspring[i].fitness.values
        else:
            for i in range(len(offspring)):
                new_parents_list = deepcopy(tools.selRandom(new_parents, 1))
                off = toolbox.mutateL(new_parents_list[0])

                offspring[i][:] = off[0]
                del offspring[i].fitness.values

        print('gens_{} crossover and mutation successfully'.format(n))
        print('gens_{} mutation successfully'.format(n))

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        train_list = find_train_inds(invalid_ind)
        print('gens_{} train individuals is:'.format(n), train_list)
        print('train individuals code are:', invalid_ind[:])
        metrics = train_population_parr(train_list=train_list, gen_num=n, population=invalid_ind, batch_size=batch_size,
                                        devices=devices, epochs=epochs, exp_name=exp_name,
                                        train_set_name=train_set_name,
                                        valid_set_name=valid_set_name, train_set_root=train_set_root,
                                        valid_set_root=valid_set_root, optimizer_name=optimizer_name,
                                        learning_rate=learning_rate,
                                        model_settings=model_settings, l2_weight_decay=l2_weight_decay)
        print('fitness of all trained model:', metrics)

        for i in range(len(offspring)):
            fitness = []
            for opt_obj in optimization_objects:
                fitness.append(metrics[i][opt_obj])
            invalid_ind[i].fitness.values = fitness

        cad_pop = population + offspring
        best5_pop = tools.selBest(cad_pop, 5)
        for ind in best5_pop:
            cad_pop.remove(ind)
        other_pop = tools.selTournament(cad_pop, k=parents_num - 5, tournsize=2)
        new_offspring = best5_pop + other_pop
                        
        sum_writer.add_scalar('best_fitness', tools.selBest(new_offspring,k=1)[0].fitness.values[0], g)
        population[:] = new_offspring
        save_population_ckpt(population=population, exp_name=exp_name, g=n)

        print('evaluate gens_{} successfully'.format(n))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

    best_ind = tools.selBest(population, parents_num)
    best_inddividuals = deepcopy(best_ind[:])
    pickle_file = open(
        os.path.join(os.path.abspath('.'), 'exps/{}/pickle/gens_{} best_individuals_code.pkl'.format(exp_name, gens)),
        'wb')
    pickle.dump(best_inddividuals, pickle_file)
    pickle_file.close()


if __name__ == '__main__':
    evolve()

import numpy
from torch.utils.data import DataLoader

from tqdm import tqdm
from loss.FocalLoss import FocalLossForSigmoid
import torch
from metrics.calculate_metrics import calculate_metrics
import shutil
from metrics.average_meter import AverageMeter
import torch.multiprocessing
from torch.nn.utils.clip_grad import clip_grad_norm_
import os
import sys
import numpy as np
import random
from thop import profile

from .util.get_optimizer import get_optimizer
from dataset.util.get_datasets import get_datasets
import multiprocessing as mp


sys.path.append('../')


def train_one_model(optimizer_name, learning_rate, l2_weight_decay, gen_num, ind_num, model, batch_size, epochs, device,
                    train_set_name, valid_set_name,
                    train_set_root, valid_set_root, exp_name,
                    mode='train'):

    seed = 12
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    model.to(device)
    model.train()

    loss_func = FocalLossForSigmoid(reduction='mean').to(device)
    optimizer = get_optimizer(optimizer_name, filter(lambda p: p.requires_grad, model.parameters()), learning_rate, l2_weight_decay)

    train_set, num_return = get_datasets(train_set_name, train_set_root, True)
    valid_set, _ = get_datasets(valid_set_name, valid_set_root, False)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=3)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)

    best_f1_score = 0
    flag = 0
    count = 0

    valid_epoch = 80
    metrics_name = ['flops', 'param', 'accuracy', 'recall', 'specificity', 'precision', 'f1_score', 'auroc', 'iou']
    metrics = {}
    for metric_name in metrics_name:
        if metric_name == 'flops' or metric_name == 'param':
            metrics.update({metric_name: 100})
        else:
            metrics.update({metric_name: 0})

    try:
        for i in range(epochs):
            train_tqdm_batch = tqdm(iterable=train_loader, total=numpy.ceil(len(train_set) / batch_size))

            for images, targets in train_tqdm_batch:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                preds = model(images)
                loss = loss_func(preds, targets)
                loss.backward()
                clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
            train_tqdm_batch.close()

            print('gens_{} individual_{}_epoch_{} train end'.format(gen_num, ind_num, i))

            epoch_acc = AverageMeter()
            epoch_recall = AverageMeter()
            epoch_precision = AverageMeter()
            epoch_specificity = AverageMeter()
            epoch_f1_score = AverageMeter()
            epoch_iou = AverageMeter()
            epoch_auroc = AverageMeter()

            if (i >= valid_epoch):
                with torch.no_grad():
                    model.eval()
                    valid_tqdm_batch = tqdm(iterable=valid_loader, total=numpy.ceil(len(valid_set) / 1))
                    
                    for images, targets in valid_tqdm_batch:
                        images = images.to(device)
                        targets = targets.to(device)
                        preds = model(images)

                        (acc, recall, specificity, precision,
                         f1_score, iou, auroc) = calculate_metrics(preds=preds, targets=targets, device=device)
                        epoch_acc.update(acc)
                        epoch_recall.update(recall)
                        epoch_precision.update(precision)
                        epoch_specificity.update(specificity)
                        epoch_f1_score.update(f1_score)
                        epoch_iou.update(iou)
                        epoch_auroc.update(auroc)

                    if i == valid_epoch:
                        flops, param = profile(model=model, inputs=(images,), verbose=False)
                        flops = flops / 1e11
                        param = param / 1e6
                  
                    print('gens_{} individual_{}_epoch_{} validate end'.format(gen_num, ind_num, i))
                    print('acc:{} | recall:{} | spe:{} | pre:{} | f1_score:{} | auroc:{}'
                          .format(epoch_acc.val,
                                  epoch_recall.val,
                                  epoch_specificity.val,
                                  epoch_precision.val,
                                  epoch_f1_score.val,
                                  epoch_auroc.val))
                    if epoch_f1_score.val > best_f1_score:
                        best_f1_score = epoch_f1_score.val

                        flag = i
                        count = 0
                        for key in list(metrics):
                            if key == 'flops':
                                metrics[key] = flops
                            elif key == 'param':
                                metrics[key] = param
                            elif key == 'accuracy':
                                metrics[key] = epoch_acc.val
                            elif key == 'recall':
                                metrics[key] = epoch_recall.val
                            elif key == 'specificity':
                                metrics[key] = epoch_specificity.val
                            elif key == 'precision':
                                metrics[key] = epoch_precision.val
                            elif key == 'f1_score':
                                metrics[key] = epoch_f1_score.val
                            elif key == 'auroc':
                                metrics[key] = epoch_auroc.val
                            elif key == 'iou':
                                metrics[key] = epoch_iou.val
                            else:
                                raise NotImplementedError

                        import pandas as pd
                        from os.path import join
                        performance_df = pd.DataFrame(
                            data=[[gen_num, ind_num, epoch_acc.val, epoch_recall.val, epoch_specificity.val,
                                   epoch_precision.val,
                                   epoch_f1_score.val, epoch_iou.val, epoch_auroc.val]],
                            columns=['epoch', 'individual', 'acc', 'recall',
                                     'specificity', 'precision', 'f1_score', 'iou',
                                     'auroc', ]

                        )
                        performance_csv_path = join(os.path.abspath('.'), 'exps/{}/csv'.format(exp_name),
                                                    'gens_{} individual_{} performance.csv'.format(gen_num, ind_num))
                        performance_df.to_csv(performance_csv_path)
                    else:
                        if i >= valid_epoch:
                            count += 1

                    end = None
                    if i > valid_epoch + 15 and best_f1_score < 0.50:
                        end = True
                    if (count >= 70) or end:
                        print('current best epoch_{} best_f1_score:'.format(flag), best_f1_score)
                        print('gens_{} individual_{} train early stop'.format(gen_num, ind_num))
                        print('=======================================================================')
                        valid_tqdm_batch.close()
                        return metrics, True
                    print('current best epoch_{} best_f1_score:'.format(flag), best_f1_score)
                    valid_tqdm_batch.close()
        print('current best epoch_{} best_f1_score:'.format(flag), best_f1_score)
        print('=======================================================================')
    except RuntimeError as exception:
        images.detach_()
        del images
        del model
        del targets
        return metrics, False
    return metrics, True

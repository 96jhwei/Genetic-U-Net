"""
Binary confusion matrix
"""

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from torch.nn import functional as F
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_binary_confusion_matrix(input_, target, device, pixel = None, threshold=0.5,
                                reduction='sum'):
    """
    Get binary confusion matrix

    Arguments:
        preds (torch tensor): raw probability outrue_positiveuts
        targets (torch tensor): ground truth
        threshold: (float): threshold value, default: 0.5
        reduction (string): either 'none' or 'sum'

    Returns:
        true_positive (torch tensor): true positive
        false_positive (torch tensor): false positive
        true_negative (torch tensor): true negative
        false_negative (torch tensor): true negative

    """
    if not input_.shape == target.shape:
        raise ValueError

    if not ((target.max() == 1.0 and target.min() == 0.0 and(target.unique().numel() == 2)) 
        or (target.max() == 0.0 and target.min() == 0.0 and(target.unique().numel() == 1))):
        raise ValueError('{}, {}, {}'.format(target.max(),target.min(),target.unique().numel()))

    input_threshed = input_.clone()
    input_threshed[input_ < threshold] = 0.0
    input_threshed[input_ >= threshold] = 1.0
    
    target_neg = -1.0 * (target - 1.0)
    input_threshed_neg = -1.0 * (input_threshed - 1.0)
    
    if pixel == None:
        
        true_positive = target * input_threshed
        false_positive = target_neg * input_threshed
    
        true_negative = target_neg * input_threshed_neg
        false_negative = target * input_threshed_neg
    else:
        kernel = torch.ones(1,1,2*pixel+1,2*pixel+1).to(device)
        target_dilation = F.conv2d(target, kernel, stride = 1, padding = pixel)
        target_dilation[target_dilation > 0] = 1
        

        true_positive = target_dilation * input_threshed
        false_positive = input_threshed - true_positive
    
        true_negative = target_neg * input_threshed_neg
        false_negative = target * input_threshed_neg
        
    if reduction == 'none':
        pass

    elif reduction == 'sum':
        true_positive = torch.sum(true_positive)
        false_positive = torch.sum(false_positive)
        true_negative = torch.sum(true_negative)
        false_negative = torch.sum(false_negative)

    return true_positive, false_positive, true_negative, false_negative

def get_threshold_binary_confusion_matrix(input_, target, device, pixel = None, reduction='sum'):
    """
    Get binary confusion matrix

    Arguments:
        preds (torch tensor): raw probability outrue_positiveuts
        targets (torch tensor): ground truth
        threshold: (float): threshold value, default: 0.5
        reduction (string): either 'none' or 'sum'

    Returns:
        true_positive (torch tensor): true positive
        false_positive (torch tensor): false positive
        true_negative (torch tensor): true negative
        false_negative (torch tensor): true negative

    """
    if not input_.shape == target.shape:
        raise ValueError

    if not ((target.max() == 1.0 and target.min() == 0.0 and(target.unique().numel() == 2)) 
        or (target.max() == 0.0 and target.min() == 0.0 and(target.unique().numel() == 1))):
        raise ValueError('{}, {}, {}'.format(target.max(),target.min(),target.unique().numel()))
        
    fusion_mat = torch.empty(0).to(device)
    for i in range(1,100):
        threshold = i/100
        input_threshed = input_.clone()
        input_threshed[input_ < threshold] = 0.0
        input_threshed[input_ >= threshold] = 1.0
    
        target_neg = -1.0 * (target - 1.0)
        input_threshed_neg = -1.0 * (input_threshed - 1.0)
    
           
        true_negative_mat = target_neg * input_threshed_neg
        false_negative_mat = target * input_threshed_neg
        
        
        if reduction == 'none':
            pass
    
        elif reduction == 'sum':
            
            true_negative = torch.sum(true_negative_mat)
            false_negative = torch.sum(false_negative_mat)
        
        if pixel == None:
            true_positive_mat = target * input_threshed
            false_positive_mat = target_neg * input_threshed 
            true_positive = torch.sum(true_positive_mat)
            false_positive = torch.sum(false_positive_mat)
        else:
            kernel = torch.ones(1,1,2*pixel+1,2*pixel+1).to(device)
            target_dilation = F.conv2d(target, kernel, stride = 1, padding = pixel)
            target_dilation[target_dilation > 0] = 1

            true_positive = torch.sum(target_dilation * input_threshed)
#            if torch.sum(input_threshed).item()>true_positive.item():
            false_positive = torch.sum(input_threshed) - true_positive
#            else:
#                false_positive= torch.tensor(0.0).to(device)
        mat = torch.stack((true_positive,false_positive,true_negative,false_negative),0)
        mat = mat.expand(1,4)
        fusion_mat = torch.cat((fusion_mat,mat),0)

    return fusion_mat
#    return true_positive, false_positive, true_negative, false_negative

if __name__ == '__main__':
    predict = Image.open('prediction-00.png','r').convert('L') 

    target = Image.open('target-00.png').convert('1')
    input_ = TF.to_tensor(predict)
    target = TF.to_tensor(target)
    
    predict_1 = Image.open('prediction-01.png').convert('L')
    
    target_1 = Image.open('target-01.png').convert('1')

    input_1 = TF.to_tensor(predict_1)

    target_1 = TF.to_tensor(target_1)
    input_ = input_.expand(1,input_.size(0),input_.size(1),input_.size(2))
    target = target.expand(1,target.size(0),target.size(1),target.size(2))
    input_1 = input_1.expand(1,input_1.size(0),input_1.size(1),input_1.size(2))
    target_1= target_1.expand(1,target_1.size(0),target_1.size(1),target_1.size(2))
    print(input_.size())
    print(target.size())
    device = torch.device('cuda:0')
    #print(device)
    #true_positive, false_positive, true_negative, false_negative = get_binary_confusion_matrix(input_,target)
    #print(true_positive)

    fusion_mat = get_threshold_binary_confusion_matrix(input_,target,device,pixel=2)
    fusion_mat_1 = get_threshold_binary_confusion_matrix(input_1,target_1,device,pixel=2)
    print(fusion_mat.size())
    fusion_mat = fusion_mat.expand(1,99,4)
    fusion_mat_1 = fusion_mat_1.expand(1,99,4)
    print(fusion_mat.size())
    fusion_mat_2 = torch.cat((fusion_mat_1,fusion_mat),0)
    print(fusion_mat_2.size())
    
    mat = fusion_mat_2.numpy()
    true_positive_s, false_positive_s, true_negative_s, false_negative_s = mat[:,:,0],mat[:,:,1],mat[:,:,2],mat[:,:,3]
    f1_per_image = (2 * true_positive_s) / (2 * true_positive_s +
                                    false_positive_s + false_negative_s)
    
    f1_max_per_image = f1_per_image.max(axis=1)
    OIS = f1_max_per_image.mean()
    f1_mean_per_image = f1_per_image.mean(axis=0)
    OIS_2 = f1_mean_per_image.max()
    mat_1 = mat.sum(axis=0)
    true_positive, false_positive, true_negative, false_negative = mat_1[:,0],mat_1[:,1],mat_1[:,2],mat_1[:,3]
    f1_all_image = (2 * true_positive) / (2 * true_positive +
                                    false_positive + false_negative)
    prc = true_positive / (true_positive + false_positive)
    acc = (true_positive + true_negative) / (
            true_positive + true_negative + false_positive + false_negative)
    iou = true_positive / (true_positive + false_positive + false_negative)
    
    AP = prc.mean()
    AIU = iou.mean()
    ODS = f1_all_image.max()
    acc_m = acc.mean()
    print(AIU,ODS,OIS,AP,acc_m,OIS_2)


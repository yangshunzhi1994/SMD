from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import shutil
import numpy as np
import torch
import time

import sys
import math
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import warnings
import losses
from torch.autograd import Variable

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        #print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_ACC_evaluation(conf_mat_a, conf_mat_b, outputs, targets_a, targets_b, lam, NUM_CLASSES=None):
    
    conf_mat_a += losses.confusion_matrix(outputs, targets_a, NUM_CLASSES)
    acc_a = sum([conf_mat_a[i, i] for i in range(conf_mat_a.shape[0])])/conf_mat_a.sum()
    precision_a = np.array([conf_mat_a[i, i]/(conf_mat_a[i].sum() + 1e-10) for i in range(conf_mat_a.shape[0])])
    recall_a = np.array([conf_mat_a[i, i]/(conf_mat_a[:, i].sum() + 1e-10) for i in range(conf_mat_a.shape[0])])
    mAP_a = sum(precision_a)/len(precision_a)
    F1_score_a = (2 * precision_a*recall_a/(precision_a+recall_a + 1e-10)).mean()

    conf_mat_b += losses.confusion_matrix(outputs, targets_b, NUM_CLASSES)
    acc_b = sum([conf_mat_b[i, i] for i in range(conf_mat_b.shape[0])])/conf_mat_b.sum()
    precision_b = np.array([conf_mat_b[i, i]/(conf_mat_b[i].sum() + 1e-10) for i in range(conf_mat_b.shape[0])])
    recall_b = np.array([conf_mat_b[i, i]/(conf_mat_b[:, i].sum() + 1e-10) for i in range(conf_mat_b.shape[0])])
    mAP_b = sum(precision_b)/len(precision_b)
    F1_score_b = (2 * precision_b*recall_b/(precision_b+recall_b + 1e-10)).mean()

    acc = lam * acc_a  +  (1 - lam) * acc_b
    mAP = lam * mAP_a  +  (1 - lam) * mAP_b
    F1_score = lam * F1_score_a  +  (1 - lam) * F1_score_b
    
    return conf_mat_a, conf_mat_b, acc, mAP, F1_score

def ACC_evaluation(conf_mat, outputs, targets, NUM_CLASSES=None):
    
    conf_mat += losses.confusion_matrix(outputs, targets, NUM_CLASSES)
    acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
    precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
    mAP = sum(precision)/len(precision)

    recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = 2 * precision*recall/(precision+recall + 1e-10)
    F1_score = f1.mean()
    
    return conf_mat, acc, mAP, F1_score

def count_parameters_in_MB(model):
	return sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def load_pretrained_model(model, pretrained_dict):
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict) 
	# 3. load the new state dict
	model.load_state_dict(model_dict)
    
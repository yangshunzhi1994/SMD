#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import itertools
import os
import time
import math
import argparse
import utils
from utils import load_pretrained_model
from datasets.RAF import RAF_multi_teacher
from datasets.PET import PET_multi_teacher
from datasets.colorferet import colorferet_multi_teacher
from datasets.FairFace import FairFace_multi_teacher
from torch.autograd import Variable
from network.teacherNet import Teacher
from tensorboardX import SummaryWriter
from utils import ACC_evaluation
from network.real_KW import real_KW2,real_KW3,real_KW4,real_KW5,real_KW6,real_KW7,real_KW8
from network.num_teacher import KW_Variance2,KW_Variance3,KW_Variance4,KW_Variance5,KW_Variance6,KW_Variance7,KW_Variance8

from utils import load_pretrained_model, count_parameters_in_MB
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--data_name', type=str, default='RAF', help='RAF,FairFace,colorferet,PET') 
parser.add_argument('--test_bs', default=200, type=int, help='learning rate')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')#1e-4,5e-4
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--number_teacher', default=4, type=int, help='Batch size')
parser.add_argument('--root', type=str, default='results/colorferet_MultiTeacher_Average', help='models and logs are saved here')

args, unparsed = parser.parse_known_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
else:
    pass

best_acc = 0
best_mAP = 0
best_F1 = 0
if args.data_name == 'colorferet':
    NUM_CLASSES = 994
elif args.data_name == 'PET':
    NUM_CLASSES = 37
else:
    NUM_CLASSES = 7

print ('The dataset used for training is:   '+ str(args.data_name))
print ('The path of the multi-teacher model is :   '+ str(args.root))
tcheckpoint = torch.load(os.path.join(args.root,'Best_MultiTeacher_model.t7'))

tnet1 = Teacher(num_classes=NUM_CLASSES).cuda()
tnet2 = Teacher(num_classes=NUM_CLASSES).cuda()
tnet3 = Teacher(num_classes=NUM_CLASSES).cuda()
tnet4 = Teacher(num_classes=NUM_CLASSES).cuda()
# tnet5 = Teacher().cuda()
# tnet6 = Teacher().cuda()
# tnet7 = Teacher().cuda()
# tnet8 = Teacher().cuda()
load_pretrained_model(tnet1, tcheckpoint['Teacher1'])
load_pretrained_model(tnet2, tcheckpoint['Teacher2'])
load_pretrained_model(tnet3, tcheckpoint['Teacher3'])
load_pretrained_model(tnet4, tcheckpoint['Teacher4'])
# load_pretrained_model(tnet5, tcheckpoint['Teacher5'])
# load_pretrained_model(tnet6, tcheckpoint['Teacher6'])
# load_pretrained_model(tnet7, tcheckpoint['Teacher7'])
# load_pretrained_model(tnet8, tcheckpoint['Teacher8'])
print ('best_Teacher1_acc is '+ str(tcheckpoint['test_Teacher1_accuracy']))
print ('best_Teacher2_acc is '+ str(tcheckpoint['test_Teacher2_accuracy']))
print ('best_Teacher3_acc is '+ str(tcheckpoint['test_Teacher3_accuracy']))
print ('best_Teacher4_acc is '+ str(tcheckpoint['test_Teacher4_accuracy']))
# print ('best_Teacher5_acc is '+ str(tcheckpoint['test_Teacher5_accuracy']))
# print ('best_Teacher6_acc is '+ str(tcheckpoint['test_Teacher6_accuracy']))
# print ('best_Teacher7_acc is '+ str(tcheckpoint['test_Teacher7_accuracy']))
# print ('best_Teacher8_acc is '+ str(tcheckpoint['test_Teacher8_accuracy']))

print ('best_Teacher_Avg_accuracy is '+ str(tcheckpoint['test_Avg_accuracy'])) 
print ('best_Teacher_Avg_MAP is '+ str(tcheckpoint['test_Avg_MAP'])) 
print ('best_Teacher_Avg_F1 is '+ str(tcheckpoint['test_Avg_F1'])) 

criterion = torch.nn.CrossEntropyLoss().cuda()

if args.data_name == 'RAF':
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.589667, 0.45717254, 0.40727714], std=[0.25235596, 0.23242524, 0.23155019])
               (transforms.ToTensor()(crop)) for crop in crops])),])
elif args.data_name == 'PET':
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.486185, 0.45276144, 0.39575183], std=[0.2640792, 0.25989106, 0.26799843])
               (transforms.ToTensor()(crop)) for crop in crops])),])
elif args.data_name == 'FairFace':
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.49167913, 0.36098105, 0.30529523], std=[0.24649838, 0.21503104, 0.20875944])
               (transforms.ToTensor()(crop)) for crop in crops])),])
elif args.data_name == 'colorferet':
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.49930117, 0.43744352, 0.37612754], std=[0.22151423, 0.24302939, 0.2520711])
               (transforms.ToTensor()(crop)) for crop in crops])),])
else:
    raise Exception('Invalid ...')

if args.data_name == 'RAF':
    PrivateTestset = RAF_multi_teacher(split = 'PrivateTest', transform=transform_test)
elif args.data_name == 'PET':
    PrivateTestset = PET_multi_teacher(split = 'PrivateTest', transform=transform_test)
elif args.data_name == 'FairFace':
    PrivateTestset = FairFace_multi_teacher(split = 'PrivateTest', transform=transform_test)
elif args.data_name == 'colorferet':
    PrivateTestset = colorferet_multi_teacher(split = 'PrivateTest', transform=transform_test)
else:
    raise Exception('Invalid dataset name...')
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=args.test_bs, shuffle=False, num_workers=1)
criterion = nn.CrossEntropyLoss()

def test(epoch):
    
    tnet1.eval()
    tnet2.eval()
    tnet3.eval()
    tnet4.eval()
#     tnet5.eval()
#     tnet6.eval()
#     tnet7.eval()
#     tnet8.eval()
    conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
#     conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
#     conf_mat6 = np.zeros((NUM_CLASSES, NUM_CLASSES))
#     conf_mat7 = np.zeros((NUM_CLASSES, NUM_CLASSES))
#     conf_mat8 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    PrivateTest_loss = 0
    Real_KW_Variance = None
    KW_Variance = []

    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        t = time.time()
        test_bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
            _, _, _, _, outputs1 = tnet1(inputs)
            _, _, _, _, outputs2 = tnet2(inputs)
            _, _, _, _, outputs3 = tnet3(inputs)
            _, _, _, _, outputs4 = tnet4(inputs)
#             _, _, _, _, outputs5 = tnet5(inputs)
#             _, _, _, _, outputs6 = tnet6(inputs)
#             _, _, _, _, outputs7 = tnet7(inputs)
#             _, _, _, _, outputs8 = tnet8(inputs)
            outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
            outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1)
            outputs3 = outputs3.view(test_bs, ncrops, -1).mean(1)
            outputs4 = outputs4.view(test_bs, ncrops, -1).mean(1)
#             outputs5 = outputs5.view(test_bs, ncrops, -1).mean(1)
#             outputs6 = outputs6.view(test_bs, ncrops, -1).mean(1)
#             outputs7 = outputs7.view(test_bs, ncrops, -1).mean(1)
#             outputs8 = outputs8.view(test_bs, ncrops, -1).mean(1)
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
            conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
            conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
            conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
#             conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
#             conf_mat6, acc6, mAP6, F1_score6 = ACC_evaluation(conf_mat6, outputs6, targets, NUM_CLASSES)
#             conf_mat7, acc7, mAP7, F1_score7 = ACC_evaluation(conf_mat7, outputs7, targets, NUM_CLASSES)
#             conf_mat8, acc8, mAP8, F1_score8 = ACC_evaluation(conf_mat8, outputs8, targets, NUM_CLASSES)
            conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, mimic, targets, NUM_CLASSES)
            Real_KW_Variance = real_KW4().cuda()(outputs1,outputs2,outputs3,outputs4,Real_KW_Variance)
            KW_Variance = KW_Variance4(outputs1,outputs2,outputs3,outputs4,targets,KW_Variance)
    
    Our_KW = Real_KW_Variance.mean()
    KW = 0   
    for i in range(0, len(KW_Variance)):
        KW = KW + KW_Variance[i]*(4-KW_Variance[i])
    KW = torch.div(KW.float(), 4*4*len(KW_Variance)).float()
    
    print("The KW of the ensemble teacher network is: %0.4f" % KW)
    print("The real KW of the ensemble teacher network is: %0.4f" % Our_KW)
    print("Test_Teacher1_accuracy: %0.5f" % acc1)
    print("Test_Teacher2_accuracy: %0.5f" % acc2)
    print("Test_Teacher3_accuracy: %0.5f" % acc3)
    print("Test_Teacher4_accuracy: %0.5f" % acc4)
#     print("Test_Teacher5_accuracy: %0.5f" % acc5)
#     print("Test_Teacher6_accuracy: %0.5f" % acc6)
#     print("Test_Teacher7_accuracy: %0.5f" % acc7)
#     print("Test_Teacher8_accuracy: %0.5f" % acc8)
    print("Test_Avg_accuracy: %0.5f" % acc) 
    print("Test_Avg_MAP: %0.5f" % mAP)
    print("Test_Avg_F1: %0.5f" % F1_score)

test_loss, test_acc, test_mAP, test_F1 = test(1)
writer.close()



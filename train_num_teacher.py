'''Train RAF/ExpW with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

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
import losses
import other
from utils import load_pretrained_model
from datasets.RAF import RAF_multi_teacher
from datasets.PET import PET_multi_teacher
from datasets.colorferet import colorferet_multi_teacher
from datasets.FairFace import FairFace_multi_teacher
from torch.autograd import Variable
from network.teacherNet import Teacher
from tensorboardX import SummaryWriter
from utils import ACC_evaluation

parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--model', type=str, default="MultiTeacher", help='MultiTeacher')
parser.add_argument('--data_name', type=str, default="RAF", help='RAF,FairFace,colorferet,PET')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='Batch size')
parser.add_argument('--test_bs', default=64, type=int, help='Batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--direction', default=0, type=float, help='direction')
parser.add_argument('--variance', default=0, type=float, help='variance')
parser.add_argument('--fusion', type=str, default="OurDiversity", help='OurDiversity')
parser.add_argument('--number_teacher', default=2, type=int, help='Batch size')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_ACC = 0
NUM_CLASSES = 7
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

total_epoch = args.epochs

path = os.path.join(args.save_root + args.data_name + '_' + args.model+ '_' + args.fusion+ '_NumberTeacher_' + str(args.number_teacher))
writer = SummaryWriter(log_dir=path)

# Data
print ('The dataset used for training is:                '+ str(args.data_name))
print ('The training mode is:                        '+ str(args.model))
print ('The fusion method is:                        '+ str(args.fusion))
print ('The direction is:                          '+ str(args.direction))
print ('The variance is:                           '+ str(args.variance))
print ('The number of teachers is:                    '+ str(args.number_teacher))
print('==> Preparing data..')

if args.data_name == 'RAF':
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5884594, 0.45767313, 0.40865755), 
                      (0.25717735, 0.23602168, 0.23505741)),
    ]) 
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.589667, 0.45717254, 0.40727714], std=[0.25235596, 0.23242524, 0.23155019])
               (transforms.ToTensor()(crop)) for crop in crops])),])
elif args.data_name == 'PET':
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.47950855, 0.4454716, 0.3953508), 
                      (0.26221144, 0.25676072, 0.2640482)),])
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.486185, 0.45276144, 0.39575183], std=[0.2640792, 0.25989106, 0.26799843])
               (transforms.ToTensor()(crop)) for crop in crops])),])
elif args.data_name == 'colorferet':
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.50150657, 0.4387828, 0.37715995), 
                      (0.22249317, 0.24526535, 0.25831717)),
    ]) 
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.49930117, 0.43744352, 0.37612754], std=[0.22151423, 0.24302939, 0.2520711])
               (transforms.ToTensor()(crop)) for crop in crops])),])
elif args.data_name == 'FairFace':
    transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4911152, 0.36028033, 0.30489963), 
                      (0.25160596, 0.21829675, 0.21198231)),
    ]) 
    transform_test = transforms.Compose([
        transforms.TenCrop(92),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.49167913, 0.36098105, 0.30529523], std=[0.24649838, 0.21503104, 0.20875944])
               (transforms.ToTensor()(crop)) for crop in crops])),])
else:
    raise Exception('Invalid ...')

if args.data_name == 'RAF':
    trainset = RAF_multi_teacher(split = 'Training', transform=transform_train)
    PrivateTestset = RAF_multi_teacher(split = 'PrivateTest', transform=transform_test)
elif args.data_name == 'PET':
    trainset = PET_multi_teacher(split = 'Training', transform=transform_train)
    PrivateTestset = PET_multi_teacher(split = 'PrivateTest', transform=transform_test)
elif args.data_name == 'colorferet':
    trainset = colorferet_multi_teacher(split = 'Training', transform=transform_train)
    PrivateTestset = colorferet_multi_teacher(split = 'PrivateTest', transform=transform_test)
elif args.data_name == 'FairFace':
    trainset = FairFace_multi_teacher(split = 'Training', transform=transform_train)
    PrivateTestset = FairFace_multi_teacher(split = 'PrivateTest', transform=transform_test)
else:
    raise Exception('Invalid dataset name...')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=args.test_bs, shuffle=False, num_workers=args.num_workers)
criterion = nn.CrossEntropyLoss()

if args.number_teacher == 2:
    net1 = Teacher().cuda()
    net2 = Teacher().cuda()
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.number_teacher == 3:
    net1 = Teacher().cuda()
    net2 = Teacher().cuda()
    net3 = Teacher().cuda()
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters(),net3.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.number_teacher == 4:
    net1 = Teacher().cuda()
    net2 = Teacher().cuda()
    net3 = Teacher().cuda()
    net4 = Teacher().cuda()
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters(),net3.parameters(),\
                                          net4.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.number_teacher == 5:
    net1 = Teacher().cuda()
    net2 = Teacher().cuda()
    net3 = Teacher().cuda()
    net4 = Teacher().cuda()
    net5 = Teacher().cuda()
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters(),net3.parameters(),net4.parameters(),\
                                          net5.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.number_teacher == 6:
    net1 = Teacher().cuda()
    net2 = Teacher().cuda()
    net3 = Teacher().cuda()
    net4 = Teacher().cuda()
    net5 = Teacher().cuda()
    net6 = Teacher().cuda()
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters(),net3.parameters(),net4.parameters(),\
                                net5.parameters(),net6.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.number_teacher == 7:
    net1 = Teacher().cuda()
    net2 = Teacher().cuda()
    net3 = Teacher().cuda()
    net4 = Teacher().cuda()
    net5 = Teacher().cuda()
    net6 = Teacher().cuda()
    net7 = Teacher().cuda()
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters(),net3.parameters(),net4.parameters(),\
                                net5.parameters(),net6.parameters(),net7.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.number_teacher == 8:
    net1 = Teacher().cuda()
    net2 = Teacher().cuda()
    net3 = Teacher().cuda()
    net4 = Teacher().cuda()
    net5 = Teacher().cuda()
    net6 = Teacher().cuda()
    net7 = Teacher().cuda()
    net8 = Teacher().cuda()
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters(),net3.parameters(),net4.parameters(),\
                   net5.parameters(),net6.parameters(),net7.parameters(),net8.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    raise Exception('Invalid ...')

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    if args.number_teacher == 2:
        net1.train()
        net2.train()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 3:
        net1.train()
        net2.train()
        net3.train()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 4:
        net1.train()
        net2.train()
        net3.train()
        net4.train()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 5:
        net1.train()
        net2.train()
        net3.train()
        net4.train()
        net5.train()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 6:
        net1.train()
        net2.train()
        net3.train()
        net4.train()
        net5.train()
        net6.train()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat6 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 7:
        net1.train()
        net2.train()
        net3.train()
        net4.train()
        net5.train()
        net6.train()
        net7.train()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat6 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat7 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 8:
        net1.train()
        net2.train()
        net3.train()
        net4.train()
        net5.train()
        net6.train()
        net7.train()
        net8.train()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat6 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat7 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat8 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    else:
        raise Exception('Invalid ...')
    train_loss = 0

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = args.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = args.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        inputs, targets = Variable(inputs), Variable(targets)
        
        if args.number_teacher == 2:
            _, _, _, _, outputs1 = net1(inputs)
            _, _, _, _, outputs2 = net2(inputs)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            mimic = (outputs1+outputs2)/2
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2
            conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
            conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
        elif args.number_teacher == 3:
            _, _, _, _, outputs1 = net1(inputs)
            _, _, _, _, outputs2 = net2(inputs)
            _, _, _, _, outputs3 = net3(inputs)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            mimic = (outputs1+outputs2+outputs3)/3
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2 + loss3
            conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
            conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
            conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
        elif args.number_teacher == 4:
            _, _, _, _, outputs1 = net1(inputs)
            _, _, _, _, outputs2 = net2(inputs)
            _, _, _, _, outputs3 = net3(inputs)
            _, _, _, _, outputs4 = net4(inputs)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            loss4 = criterion(outputs4, targets)
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2 + loss3 + loss4
            conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
            conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
            conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
            conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
        elif args.number_teacher == 5:
            _, _, _, _, outputs1 = net1(inputs)
            _, _, _, _, outputs2 = net2(inputs)
            _, _, _, _, outputs3 = net3(inputs)
            _, _, _, _, outputs4 = net4(inputs)
            _, _, _, _, outputs5 = net5(inputs)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            loss4 = criterion(outputs4, targets)
            loss5 = criterion(outputs5, targets)
            mimic = (outputs1+outputs2+outputs3+outputs4+outputs5)/5
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2 + loss3 + loss4 + loss5
            conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
            conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
            conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
            conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
            conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
        elif args.number_teacher == 6:
            _, _, _, _, outputs1 = net1(inputs)
            _, _, _, _, outputs2 = net2(inputs)
            _, _, _, _, outputs3 = net3(inputs)
            _, _, _, _, outputs4 = net4(inputs)
            _, _, _, _, outputs5 = net5(inputs)
            _, _, _, _, outputs6 = net6(inputs)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            loss4 = criterion(outputs4, targets)
            loss5 = criterion(outputs5, targets)
            loss6 = criterion(outputs6, targets)
            mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6)/6
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
            conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
            conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
            conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
            conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
            conf_mat6, acc6, mAP6, F1_score6 = ACC_evaluation(conf_mat6, outputs6, targets, NUM_CLASSES)
        elif args.number_teacher == 7:
            _, _, _, _, outputs1 = net1(inputs)
            _, _, _, _, outputs2 = net2(inputs)
            _, _, _, _, outputs3 = net3(inputs)
            _, _, _, _, outputs4 = net4(inputs)
            _, _, _, _, outputs5 = net5(inputs)
            _, _, _, _, outputs6 = net6(inputs)
            _, _, _, _, outputs7 = net7(inputs)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            loss4 = criterion(outputs4, targets)
            loss5 = criterion(outputs5, targets)
            loss6 = criterion(outputs6, targets)
            loss7 = criterion(outputs7, targets)
            mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6+outputs7)/7
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
            conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
            conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
            conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
            conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
            conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
            conf_mat6, acc6, mAP6, F1_score6 = ACC_evaluation(conf_mat6, outputs6, targets, NUM_CLASSES)
            conf_mat7, acc7, mAP7, F1_score7 = ACC_evaluation(conf_mat7, outputs7, targets, NUM_CLASSES)
        elif args.number_teacher == 8:
            _, _, _, _, outputs1 = net1(inputs)
            _, _, _, _, outputs2 = net2(inputs)
            _, _, _, _, outputs3 = net3(inputs)
            _, _, _, _, outputs4 = net4(inputs)
            _, _, _, _, outputs5 = net5(inputs)
            _, _, _, _, outputs6 = net6(inputs)
            _, _, _, _, outputs7 = net7(inputs)
            _, _, _, _, outputs8 = net8(inputs)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            loss4 = criterion(outputs4, targets)
            loss5 = criterion(outputs5, targets)
            loss6 = criterion(outputs6, targets)
            loss7 = criterion(outputs7, targets)
            loss8 = criterion(outputs8, targets)
            mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6+outputs7+outputs8)/8
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8
            conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
            conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
            conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
            conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
            conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
            conf_mat6, acc6, mAP6, F1_score6 = ACC_evaluation(conf_mat6, outputs6, targets, NUM_CLASSES)
            conf_mat7, acc7, mAP7, F1_score7 = ACC_evaluation(conf_mat7, outputs7, targets, NUM_CLASSES)
            conf_mat8, acc8, mAP8, F1_score8 = ACC_evaluation(conf_mat8, outputs8, targets, NUM_CLASSES)
        else:
            raise Exception('Invalid ...')
            
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        
        conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, mimic, targets, NUM_CLASSES)
    
    if args.number_teacher == 2:
        return train_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 3:
        return train_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 4:
        return train_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 5:
        return train_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc5, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 6:
        return train_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc5, 100.*acc6, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 7:
        return train_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc5, 100.*acc6, 100.*acc7, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 8:
        return train_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc5, 100.*acc6, 100.*acc7, 100.*acc8, 100.*acc, 100.* mAP, 100 * F1_score
    else:
        raise Exception('Invalid ...')

def test(epoch):
    if args.number_teacher == 2:
        net1.eval()
        net2.eval()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 3:
        net1.eval()
        net2.eval()
        net3.eval()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 4:
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 5:
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 6:
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        net6.eval()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat6 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 7:
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        net6.eval()
        net7.eval()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat6 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat7 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    elif args.number_teacher == 8:
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        net5.eval()
        net6.eval()
        net7.eval()
        net8.eval()
        conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat5 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat6 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat7 = np.zeros((NUM_CLASSES, NUM_CLASSES))
        conf_mat8 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    else:
        raise Exception('Invalid ...')
    
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    PrivateTest_loss = 0

    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        t = time.time()
        test_bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        
        with torch.no_grad():
            if args.number_teacher == 2:
                _, _, _, _, outputs1 = net1(inputs)
                _, _, _, _, outputs2 = net2(inputs)
                outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
                outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1)
                mimic = (outputs1+outputs2)/2
                conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
                conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
            elif args.number_teacher == 3:
                _, _, _, _, outputs1 = net1(inputs)
                _, _, _, _, outputs2 = net2(inputs)
                _, _, _, _, outputs3 = net3(inputs)
                outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
                outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1)
                outputs3 = outputs3.view(test_bs, ncrops, -1).mean(1)
                mimic = (outputs1+outputs2+outputs3)/3
                conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
                conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
                conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
            elif args.number_teacher == 4:
                _, _, _, _, outputs1 = net1(inputs)
                _, _, _, _, outputs2 = net2(inputs)
                _, _, _, _, outputs3 = net3(inputs)
                _, _, _, _, outputs4 = net4(inputs)
                outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
                outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1)
                outputs3 = outputs3.view(test_bs, ncrops, -1).mean(1)
                outputs4 = outputs4.view(test_bs, ncrops, -1).mean(1)
                mimic = (outputs1+outputs2+outputs3+outputs4)/4
                conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
                conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
                conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
                conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
            elif args.number_teacher == 5:
                _, _, _, _, outputs1 = net1(inputs)
                _, _, _, _, outputs2 = net2(inputs)
                _, _, _, _, outputs3 = net3(inputs)
                _, _, _, _, outputs4 = net4(inputs)
                _, _, _, _, outputs5 = net5(inputs)
                outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
                outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1)
                outputs3 = outputs3.view(test_bs, ncrops, -1).mean(1)
                outputs4 = outputs4.view(test_bs, ncrops, -1).mean(1)
                outputs5 = outputs5.view(test_bs, ncrops, -1).mean(1)
                mimic = (outputs1+outputs2+outputs3+outputs4+outputs5)/5
                conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
                conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
                conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
                conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
                conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
            elif args.number_teacher == 6:
                _, _, _, _, outputs1 = net1(inputs)
                _, _, _, _, outputs2 = net2(inputs)
                _, _, _, _, outputs3 = net3(inputs)
                _, _, _, _, outputs4 = net4(inputs)
                _, _, _, _, outputs5 = net5(inputs)
                _, _, _, _, outputs6 = net6(inputs)
                outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
                outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1)
                outputs3 = outputs3.view(test_bs, ncrops, -1).mean(1)
                outputs4 = outputs4.view(test_bs, ncrops, -1).mean(1)
                outputs5 = outputs5.view(test_bs, ncrops, -1).mean(1)
                outputs6 = outputs6.view(test_bs, ncrops, -1).mean(1)
                mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6)/6
                conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
                conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
                conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
                conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
                conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
                conf_mat6, acc6, mAP6, F1_score6 = ACC_evaluation(conf_mat6, outputs6, targets, NUM_CLASSES)
            elif args.number_teacher == 7:
                _, _, _, _, outputs1 = net1(inputs)
                _, _, _, _, outputs2 = net2(inputs)
                _, _, _, _, outputs3 = net3(inputs)
                _, _, _, _, outputs4 = net4(inputs)
                _, _, _, _, outputs5 = net5(inputs)
                _, _, _, _, outputs6 = net6(inputs)
                _, _, _, _, outputs7 = net7(inputs)
                outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
                outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1)
                outputs3 = outputs3.view(test_bs, ncrops, -1).mean(1)
                outputs4 = outputs4.view(test_bs, ncrops, -1).mean(1)
                outputs5 = outputs5.view(test_bs, ncrops, -1).mean(1)
                outputs6 = outputs6.view(test_bs, ncrops, -1).mean(1)
                outputs7 = outputs7.view(test_bs, ncrops, -1).mean(1)
                mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6+outputs7)/7
                conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
                conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
                conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
                conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
                conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
                conf_mat6, acc6, mAP6, F1_score6 = ACC_evaluation(conf_mat6, outputs6, targets, NUM_CLASSES)
                conf_mat7, acc7, mAP7, F1_score7 = ACC_evaluation(conf_mat7, outputs7, targets, NUM_CLASSES)
            elif args.number_teacher == 8:
                _, _, _, _, outputs1 = net1(inputs)
                _, _, _, _, outputs2 = net2(inputs)
                _, _, _, _, outputs3 = net3(inputs)
                _, _, _, _, outputs4 = net4(inputs)
                _, _, _, _, outputs5 = net5(inputs)
                _, _, _, _, outputs6 = net6(inputs)
                _, _, _, _, outputs7 = net7(inputs)
                _, _, _, _, outputs8 = net8(inputs)
                outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
                outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1)
                outputs3 = outputs3.view(test_bs, ncrops, -1).mean(1)
                outputs4 = outputs4.view(test_bs, ncrops, -1).mean(1)
                outputs5 = outputs5.view(test_bs, ncrops, -1).mean(1)
                outputs6 = outputs6.view(test_bs, ncrops, -1).mean(1)
                outputs7 = outputs7.view(test_bs, ncrops, -1).mean(1)
                outputs8 = outputs8.view(test_bs, ncrops, -1).mean(1)
                mimic = (outputs1+outputs2+outputs3+outputs4+outputs5+outputs6+outputs7+outputs8)/8
                conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
                conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
                conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
                conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
                conf_mat5, acc5, mAP5, F1_score5 = ACC_evaluation(conf_mat5, outputs5, targets, NUM_CLASSES)
                conf_mat6, acc6, mAP6, F1_score6 = ACC_evaluation(conf_mat6, outputs6, targets, NUM_CLASSES)
                conf_mat7, acc7, mAP7, F1_score7 = ACC_evaluation(conf_mat7, outputs7, targets, NUM_CLASSES)
                conf_mat8, acc8, mAP8, F1_score8 = ACC_evaluation(conf_mat8, outputs8, targets, NUM_CLASSES)
            else:
                raise Exception('Invalid ...')
            
        loss = criterion(mimic, targets)
        PrivateTest_loss += loss.item()
        conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, mimic, targets, NUM_CLASSES)
    
    if args.number_teacher == 2:
        return PrivateTest_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 3:
        return PrivateTest_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 4:
        return PrivateTest_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 5:
        return PrivateTest_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc5, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 6:
        return PrivateTest_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc5, 100.*acc6, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 7:
        return PrivateTest_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc5, 100.*acc6, 100.*acc7, 100.*acc, 100.* mAP, 100 * F1_score
    elif args.number_teacher == 8:
        return PrivateTest_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc5, 100.*acc6, 100.*acc7, 100.*acc8, 100.*acc, 100.* mAP, 100 * F1_score
    else:
        raise Exception('Invalid ...')

for epoch in range(0, total_epoch):
    
    if args.number_teacher == 2:
        train_loss, train_acc1, train_acc2, train_avgACC, train_avgMAP, train_avgF1 = train(epoch)
        test_loss, test_acc1, test_acc2, test_avgACC, test_avgMAP, test_avgF1 = test(epoch)
        print("train_loss:  %0.3f, train_acc1:  %0.3f, train_acc2:  %0.3f, train_avgACC:  %0.3f, train_avgMAP:  %0.3f, train_avgF1:  %0.3f"%(train_loss, train_acc1, train_acc2, train_avgACC, train_avgMAP, train_avgF1))
        print("test_loss:  %0.3f, test_acc1:  %0.3f, test_acc2:  %0.3f, test_avgACC:  %0.3f, test_avgMAP:  %0.3f, test_avgF1:  %0.3f"%(test_loss, test_acc1, test_acc2, test_avgACC, test_avgMAP, test_avgF1))
        writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('epoch/Teacher1_accuracy', {'train': train_acc1, 'test': test_acc1}, epoch)
        writer.add_scalars('epoch/Teacher2_accuracy', {'train': train_acc2, 'test': test_acc2}, epoch)
        writer.add_scalars('epoch/Avg_accuracy', {'train': train_avgACC, 'test': test_avgACC}, epoch)
        writer.add_scalars('epoch/Avg_MAP', {'train': train_avgMAP, 'test': test_avgMAP}, epoch)
        writer.add_scalars('epoch/Avg_F1', {'train': train_avgF1, 'test': test_avgF1}, epoch)
        if test_avgACC > best_ACC:
            best_ACC = test_avgACC
            print ('Saving models......')
            print("Test_Teacher1_accuracy: %0.3f" % test_acc1)
            print("Test_Teacher2_accuracy: %0.3f" % test_acc2)
            print("Test_Avg_accuracy: %0.3f" % test_avgACC)
            print("Test_Avg_MAP: %0.3f" % test_avgMAP)
            print("Test_Avg_F1: %0.3f" % test_avgF1)
            state = {
                'Teacher1': net1.state_dict() if use_cuda else net1,
                'Teacher2': net2.state_dict() if use_cuda else net2,
                'test_Teacher1_accuracy': test_acc1,
                'test_Teacher2_accuracy': test_acc2,
                'test_Avg_accuracy': test_avgACC, 
                'test_Avg_MAP': test_avgMAP,
                'test_Avg_F1': test_avgF1, 
                'test_epoch': epoch,
            } 
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(state, os.path.join(path,'Best_MultiTeacher_model.t7'))
            
    elif args.number_teacher == 3:
        train_loss, train_acc1, train_acc2, train_acc3, train_avgACC, train_avgMAP, train_avgF1 = train(epoch)
        test_loss, test_acc1, test_acc2, test_acc3, test_avgACC, test_avgMAP, test_avgF1 = test(epoch)
        print("train_loss:  %0.3f, train_acc1:  %0.3f, train_acc2:  %0.3f, train_acc3:  %0.3f, train_avgACC:  %0.3f, train_avgMAP:  %0.3f, train_avgF1:  %0.3f"%(train_loss, train_acc1, train_acc2, train_acc3, train_avgACC, train_avgMAP, train_avgF1))
        print("test_loss:  %0.3f, test_acc1:  %0.3f, test_acc2:  %0.3f, test_acc3:  %0.3f, test_avgACC:  %0.3f, test_avgMAP:  %0.3f, test_avgF1:  %0.3f"%(test_loss, test_acc1, test_acc2, test_acc3, test_avgACC, test_avgMAP, test_avgF1))
        writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('epoch/Teacher1_accuracy', {'train': train_acc1, 'test': test_acc1}, epoch)
        writer.add_scalars('epoch/Teacher2_accuracy', {'train': train_acc2, 'test': test_acc2}, epoch)
        writer.add_scalars('epoch/Teacher3_accuracy', {'train': train_acc3, 'test': test_acc3}, epoch)
        writer.add_scalars('epoch/Avg_accuracy', {'train': train_avgACC, 'test': test_avgACC}, epoch)
        writer.add_scalars('epoch/Avg_MAP', {'train': train_avgMAP, 'test': test_avgMAP}, epoch)
        writer.add_scalars('epoch/Avg_F1', {'train': train_avgF1, 'test': test_avgF1}, epoch)
        if test_avgACC > best_ACC:
            best_ACC = test_avgACC
            print ('Saving models......')
            print("Test_Teacher1_accuracy: %0.3f" % test_acc1)
            print("Test_Teacher2_accuracy: %0.3f" % test_acc2)
            print("Test_Teacher3_accuracy: %0.3f" % test_acc3)
            print("Test_Avg_accuracy: %0.3f" % test_avgACC)
            print("Test_Avg_MAP: %0.3f" % test_avgMAP)
            print("Test_Avg_F1: %0.3f" % test_avgF1)
            state = {
                'Teacher1': net1.state_dict() if use_cuda else net1,
                'Teacher2': net2.state_dict() if use_cuda else net2,
                'Teacher3': net3.state_dict() if use_cuda else net3,
                'test_Teacher1_accuracy': test_acc1,
                'test_Teacher2_accuracy': test_acc2,
                'test_Teacher3_accuracy': test_acc3,
                'test_Avg_accuracy': test_avgACC, 
                'test_Avg_MAP': test_avgMAP,
                'test_Avg_F1': test_avgF1, 
                'test_epoch': epoch,
            } 
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(state, os.path.join(path,'Best_MultiTeacher_model.t7'))
            
    elif args.number_teacher == 4:
        train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_avgACC, train_avgMAP, train_avgF1 = train(epoch)
        test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_avgACC, test_avgMAP, test_avgF1 = test(epoch)
        print("train_loss:  %0.3f, train_acc1:  %0.3f, train_acc2:  %0.3f, train_acc3:  %0.3f, train_acc4:  %0.3f, train_avgACC:  %0.3f, train_avgMAP:  %0.3f, train_avgF1:  %0.3f"%(train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_avgACC, train_avgMAP, train_avgF1))
        print("test_loss:  %0.3f, test_acc1:  %0.3f, test_acc2:  %0.3f, test_acc3:  %0.3f, test_acc4:  %0.3f, test_avgACC:  %0.3f, test_avgMAP:  %0.3f, test_avgF1:  %0.3f"%(test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_avgACC, test_avgMAP, test_avgF1))
        writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('epoch/Teacher1_accuracy', {'train': train_acc1, 'test': test_acc1}, epoch)
        writer.add_scalars('epoch/Teacher2_accuracy', {'train': train_acc2, 'test': test_acc2}, epoch)
        writer.add_scalars('epoch/Teacher3_accuracy', {'train': train_acc3, 'test': test_acc3}, epoch)
        writer.add_scalars('epoch/Teacher4_accuracy', {'train': train_acc4, 'test': test_acc4}, epoch)
        writer.add_scalars('epoch/Avg_accuracy', {'train': train_avgACC, 'test': test_avgACC}, epoch)
        writer.add_scalars('epoch/Avg_MAP', {'train': train_avgMAP, 'test': test_avgMAP}, epoch)
        writer.add_scalars('epoch/Avg_F1', {'train': train_avgF1, 'test': test_avgF1}, epoch)
        if test_avgACC > best_ACC:
            best_ACC = test_avgACC
            print ('Saving models......')
            print("Test_Teacher1_accuracy: %0.3f" % test_acc1)
            print("Test_Teacher2_accuracy: %0.3f" % test_acc2)
            print("Test_Teacher3_accuracy: %0.3f" % test_acc3)
            print("Test_Teacher4_accuracy: %0.3f" % test_acc4)
            print("Test_Avg_accuracy: %0.3f" % test_avgACC)
            print("Test_Avg_MAP: %0.3f" % test_avgMAP)
            print("Test_Avg_F1: %0.3f" % test_avgF1)
            state = {
                'Teacher1': net1.state_dict() if use_cuda else net1,
                'Teacher2': net2.state_dict() if use_cuda else net2,
                'Teacher3': net3.state_dict() if use_cuda else net3,
                'Teacher4': net4.state_dict() if use_cuda else net4,
                'test_Teacher1_accuracy': test_acc1,
                'test_Teacher2_accuracy': test_acc2,
                'test_Teacher3_accuracy': test_acc3,
                'test_Teacher4_accuracy': test_acc4,
                'test_Avg_accuracy': test_avgACC, 
                'test_Avg_MAP': test_avgMAP,
                'test_Avg_F1': test_avgF1, 
                'test_epoch': epoch,
            } 
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(state, os.path.join(path,'Best_MultiTeacher_model.t7'))
    
    elif args.number_teacher == 5:
        train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_avgACC, train_avgMAP, train_avgF1 = train(epoch)
        test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_avgACC, test_avgMAP, test_avgF1 = test(epoch)
        print("train_loss:  %0.3f, train_acc1:  %0.3f, train_acc2:  %0.3f, train_acc3:  %0.3f, train_acc4:  %0.3f, train_acc5:  %0.3f, train_avgACC:  %0.3f, train_avgMAP:  %0.3f, train_avgF1:  %0.3f"%(train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_avgACC, train_avgMAP, train_avgF1))
        print("test_loss:  %0.3f, test_acc1:  %0.3f, test_acc2:  %0.3f, test_acc3:  %0.3f, test_acc4:  %0.3f, test_acc5:  %0.3f, test_avgACC:  %0.3f, test_avgMAP:  %0.3f, test_avgF1:  %0.3f"%(test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_avgACC, test_avgMAP, test_avgF1))
        writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('epoch/Teacher1_accuracy', {'train': train_acc1, 'test': test_acc1}, epoch)
        writer.add_scalars('epoch/Teacher2_accuracy', {'train': train_acc2, 'test': test_acc2}, epoch)
        writer.add_scalars('epoch/Teacher3_accuracy', {'train': train_acc3, 'test': test_acc3}, epoch)
        writer.add_scalars('epoch/Teacher4_accuracy', {'train': train_acc4, 'test': test_acc4}, epoch)
        writer.add_scalars('epoch/Teacher5_accuracy', {'train': train_acc5, 'test': test_acc5}, epoch)
        writer.add_scalars('epoch/Avg_accuracy', {'train': train_avgACC, 'test': test_avgACC}, epoch)
        writer.add_scalars('epoch/Avg_MAP', {'train': train_avgMAP, 'test': test_avgMAP}, epoch)
        writer.add_scalars('epoch/Avg_F1', {'train': train_avgF1, 'test': test_avgF1}, epoch)
        if test_avgACC > best_ACC:
            best_ACC = test_avgACC
            print ('Saving models......')
            print("Test_Teacher1_accuracy: %0.3f" % test_acc1)
            print("Test_Teacher2_accuracy: %0.3f" % test_acc2)
            print("Test_Teacher3_accuracy: %0.3f" % test_acc3)
            print("Test_Teacher4_accuracy: %0.3f" % test_acc4)
            print("Test_Teacher5_accuracy: %0.3f" % test_acc5)
            print("Test_Avg_accuracy: %0.3f" % test_avgACC)
            print("Test_Avg_MAP: %0.3f" % test_avgMAP)
            print("Test_Avg_F1: %0.3f" % test_avgF1)
            state = {
                'Teacher1': net1.state_dict() if use_cuda else net1,
                'Teacher2': net2.state_dict() if use_cuda else net2,
                'Teacher3': net3.state_dict() if use_cuda else net3,
                'Teacher4': net4.state_dict() if use_cuda else net4,
                'Teacher5': net5.state_dict() if use_cuda else net5,
                'test_Teacher1_accuracy': test_acc1,
                'test_Teacher2_accuracy': test_acc2,
                'test_Teacher3_accuracy': test_acc3,
                'test_Teacher4_accuracy': test_acc4,
                'test_Teacher5_accuracy': test_acc5,
                'test_Avg_accuracy': test_avgACC, 
                'test_Avg_MAP': test_avgMAP,
                'test_Avg_F1': test_avgF1, 
                'test_epoch': epoch,
            } 
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(state, os.path.join(path,'Best_MultiTeacher_model.t7'))
    
    elif args.number_teacher == 6:
        train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_acc6, train_avgACC, train_avgMAP, train_avgF1 = train(epoch)
        test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc6, test_avgACC, test_avgMAP, test_avgF1 = test(epoch)
        print("train_loss:  %0.3f, train_acc1:  %0.3f, train_acc2:  %0.3f, train_acc3:  %0.3f, train_acc4:  %0.3f, train_acc5:  %0.3f, train_acc6:  %0.3f, train_avgACC:  %0.3f, train_avgMAP:  %0.3f, train_avgF1:  %0.3f"%(train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_acc6, train_avgACC, train_avgMAP, train_avgF1))
        print("test_loss:  %0.3f, test_acc1:  %0.3f, test_acc2:  %0.3f, test_acc3:  %0.3f, test_acc4:  %0.3f, test_acc5:  %0.3f, test_acc6:  %0.3f, test_avgACC:  %0.3f, test_avgMAP:  %0.3f, test_avgF1:  %0.3f"%(test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc6, test_avgACC, test_avgMAP, test_avgF1))
        writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('epoch/Teacher1_accuracy', {'train': train_acc1, 'test': test_acc1}, epoch)
        writer.add_scalars('epoch/Teacher2_accuracy', {'train': train_acc2, 'test': test_acc2}, epoch)
        writer.add_scalars('epoch/Teacher3_accuracy', {'train': train_acc3, 'test': test_acc3}, epoch)
        writer.add_scalars('epoch/Teacher4_accuracy', {'train': train_acc4, 'test': test_acc4}, epoch)
        writer.add_scalars('epoch/Teacher5_accuracy', {'train': train_acc5, 'test': test_acc5}, epoch)
        writer.add_scalars('epoch/Teacher6_accuracy', {'train': train_acc6, 'test': test_acc6}, epoch)
        writer.add_scalars('epoch/Avg_accuracy', {'train': train_avgACC, 'test': test_avgACC}, epoch)
        writer.add_scalars('epoch/Avg_MAP', {'train': train_avgMAP, 'test': test_avgMAP}, epoch)
        writer.add_scalars('epoch/Avg_F1', {'train': train_avgF1, 'test': test_avgF1}, epoch)
        if test_avgACC > best_ACC:
            best_ACC = test_avgACC
            print ('Saving models......')
            print("Test_Teacher1_accuracy: %0.3f" % test_acc1)
            print("Test_Teacher2_accuracy: %0.3f" % test_acc2)
            print("Test_Teacher3_accuracy: %0.3f" % test_acc3)
            print("Test_Teacher4_accuracy: %0.3f" % test_acc4)
            print("Test_Teacher5_accuracy: %0.3f" % test_acc5)
            print("Test_Teacher6_accuracy: %0.3f" % test_acc6)
            print("Test_Avg_accuracy: %0.3f" % test_avgACC)
            print("Test_Avg_MAP: %0.3f" % test_avgMAP)
            print("Test_Avg_F1: %0.3f" % test_avgF1)
            state = {
                'Teacher1': net1.state_dict() if use_cuda else net1,
                'Teacher2': net2.state_dict() if use_cuda else net2,
                'Teacher3': net3.state_dict() if use_cuda else net3,
                'Teacher4': net4.state_dict() if use_cuda else net4,
                'Teacher5': net5.state_dict() if use_cuda else net5,
                'Teacher6': net6.state_dict() if use_cuda else net6,
                'test_Teacher1_accuracy': test_acc1,
                'test_Teacher2_accuracy': test_acc2,
                'test_Teacher3_accuracy': test_acc3,
                'test_Teacher4_accuracy': test_acc4,
                'test_Teacher5_accuracy': test_acc5,
                'test_Teacher6_accuracy': test_acc6,
                'test_Avg_accuracy': test_avgACC, 
                'test_Avg_MAP': test_avgMAP,
                'test_Avg_F1': test_avgF1, 
                'test_epoch': epoch,
            } 
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(state, os.path.join(path,'Best_MultiTeacher_model.t7'))
    
    elif args.number_teacher == 7:
        train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_acc6, train_acc7, train_avgACC, train_avgMAP, train_avgF1 = train(epoch)
        test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc6, test_acc7, test_avgACC, test_avgMAP, test_avgF1 = test(epoch)
        print("train_loss:  %0.3f, train_acc1:  %0.3f, train_acc2:  %0.3f, train_acc3:  %0.3f, train_acc4:  %0.3f, train_acc5:  %0.3f, train_acc6:  %0.3f, train_acc7:  %0.3f, train_avgACC:  %0.3f, train_avgMAP:  %0.3f, train_avgF1:  %0.3f"%(train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_acc6, train_acc7, train_avgACC, train_avgMAP, train_avgF1))
        print("test_loss:  %0.3f, test_acc1:  %0.3f, test_acc2:  %0.3f, test_acc3:  %0.3f, test_acc4:  %0.3f, test_acc5:  %0.3f, test_acc6:  %0.3f, test_acc7:  %0.3f, test_avgACC:  %0.3f, test_avgMAP:  %0.3f, test_avgF1:  %0.3f"%(test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc6, test_acc7, test_avgACC, test_avgMAP, test_avgF1))
        writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('epoch/Teacher1_accuracy', {'train': train_acc1, 'test': test_acc1}, epoch)
        writer.add_scalars('epoch/Teacher2_accuracy', {'train': train_acc2, 'test': test_acc2}, epoch)
        writer.add_scalars('epoch/Teacher3_accuracy', {'train': train_acc3, 'test': test_acc3}, epoch)
        writer.add_scalars('epoch/Teacher4_accuracy', {'train': train_acc4, 'test': test_acc4}, epoch)
        writer.add_scalars('epoch/Teacher5_accuracy', {'train': train_acc5, 'test': test_acc5}, epoch)
        writer.add_scalars('epoch/Teacher6_accuracy', {'train': train_acc6, 'test': test_acc6}, epoch)
        writer.add_scalars('epoch/Teacher7_accuracy', {'train': train_acc7, 'test': test_acc7}, epoch)
        writer.add_scalars('epoch/Avg_accuracy', {'train': train_avgACC, 'test': test_avgACC}, epoch)
        writer.add_scalars('epoch/Avg_MAP', {'train': train_avgMAP, 'test': test_avgMAP}, epoch)
        writer.add_scalars('epoch/Avg_F1', {'train': train_avgF1, 'test': test_avgF1}, epoch)
        if test_avgACC > best_ACC:
            best_ACC = test_avgACC
            print ('Saving models......')
            print("Test_Teacher1_accuracy: %0.3f" % test_acc1)
            print("Test_Teacher2_accuracy: %0.3f" % test_acc2)
            print("Test_Teacher3_accuracy: %0.3f" % test_acc3)
            print("Test_Teacher4_accuracy: %0.3f" % test_acc4)
            print("Test_Teacher5_accuracy: %0.3f" % test_acc5)
            print("Test_Teacher6_accuracy: %0.3f" % test_acc6)
            print("Test_Teacher7_accuracy: %0.3f" % test_acc7)
            print("Test_Avg_accuracy: %0.3f" % test_avgACC)
            print("Test_Avg_MAP: %0.3f" % test_avgMAP)
            print("Test_Avg_F1: %0.3f" % test_avgF1)
            state = {
                'Teacher1': net1.state_dict() if use_cuda else net1,
                'Teacher2': net2.state_dict() if use_cuda else net2,
                'Teacher3': net3.state_dict() if use_cuda else net3,
                'Teacher4': net4.state_dict() if use_cuda else net4,
                'Teacher5': net5.state_dict() if use_cuda else net5,
                'Teacher6': net6.state_dict() if use_cuda else net6,
                'Teacher7': net7.state_dict() if use_cuda else net7,
                'test_Teacher1_accuracy': test_acc1,
                'test_Teacher2_accuracy': test_acc2,
                'test_Teacher3_accuracy': test_acc3,
                'test_Teacher4_accuracy': test_acc4,
                'test_Teacher5_accuracy': test_acc5,
                'test_Teacher6_accuracy': test_acc6,
                'test_Teacher7_accuracy': test_acc7,
                'test_Avg_accuracy': test_avgACC, 
                'test_Avg_MAP': test_avgMAP,
                'test_Avg_F1': test_avgF1, 
                'test_epoch': epoch,
            } 
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(state, os.path.join(path,'Best_MultiTeacher_model.t7'))
    
    elif args.number_teacher == 8:
        train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_acc6, train_acc7, train_acc8, train_avgACC, train_avgMAP, train_avgF1 = train(epoch)
        test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc6, test_acc7, test_acc8, test_avgACC, test_avgMAP, test_avgF1  = test(epoch)
        print("train_loss:  %0.3f, train_acc1:  %0.3f, train_acc2:  %0.3f, train_acc3:  %0.3f, train_acc4:  %0.3f, train_acc5:  %0.3f, train_acc6:  %0.3f, train_acc7:  %0.3f, train_acc8:  %0.3f, train_avgACC:  %0.3f, train_avgMAP:  %0.3f, train_avgF1:  %0.3f"%(train_loss, train_acc1, train_acc2, train_acc3, train_acc4, train_acc5, train_acc6, train_acc7, train_acc8, train_avgACC, train_avgMAP, train_avgF1))
        print("test_loss:  %0.3f, test_acc1:  %0.3f, test_acc2:  %0.3f, test_acc3:  %0.3f, test_acc4:  %0.3f, test_acc5:  %0.3f, test_acc6:  %0.3f, test_acc7:  %0.3f, test_acc8:  %0.3f, test_avgACC:  %0.3f, test_avgMAP:  %0.3f, test_avgF1:  %0.3f"%(test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc6, test_acc7, test_acc8, test_avgACC, test_avgMAP, test_avgF1))
        writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars('epoch/Teacher1_accuracy', {'train': train_acc1, 'test': test_acc1}, epoch)
        writer.add_scalars('epoch/Teacher2_accuracy', {'train': train_acc2, 'test': test_acc2}, epoch)
        writer.add_scalars('epoch/Teacher3_accuracy', {'train': train_acc3, 'test': test_acc3}, epoch)
        writer.add_scalars('epoch/Teacher4_accuracy', {'train': train_acc4, 'test': test_acc4}, epoch)
        writer.add_scalars('epoch/Teacher5_accuracy', {'train': train_acc5, 'test': test_acc5}, epoch)
        writer.add_scalars('epoch/Teacher6_accuracy', {'train': train_acc6, 'test': test_acc6}, epoch)
        writer.add_scalars('epoch/Teacher7_accuracy', {'train': train_acc7, 'test': test_acc7}, epoch)
        writer.add_scalars('epoch/Teacher8_accuracy', {'train': train_acc8, 'test': test_acc8}, epoch)
        writer.add_scalars('epoch/Avg_accuracy', {'train': train_avgACC, 'test': test_avgACC}, epoch)
        writer.add_scalars('epoch/Avg_MAP', {'train': train_avgMAP, 'test': test_avgMAP}, epoch)
        writer.add_scalars('epoch/Avg_F1', {'train': train_avgF1, 'test': test_avgF1}, epoch)
        if test_avgACC > best_ACC:
            best_ACC = test_avgACC
            print ('Saving models......')
            print("Test_Teacher1_accuracy: %0.3f" % test_acc1)
            print("Test_Teacher2_accuracy: %0.3f" % test_acc2)
            print("Test_Teacher3_accuracy: %0.3f" % test_acc3)
            print("Test_Teacher4_accuracy: %0.3f" % test_acc4)
            print("Test_Teacher5_accuracy: %0.3f" % test_acc5)
            print("Test_Teacher6_accuracy: %0.3f" % test_acc6)
            print("Test_Teacher7_accuracy: %0.3f" % test_acc7)
            print("Test_Teacher8_accuracy: %0.3f" % test_acc8)
            print("Test_Avg_accuracy: %0.3f" % test_avgACC)
            print("Test_Avg_MAP: %0.3f" % test_avgMAP)
            print("Test_Avg_F1: %0.3f" % test_avgF1)
            state = {
                'Teacher1': net1.state_dict() if use_cuda else net1,
                'Teacher2': net2.state_dict() if use_cuda else net2,
                'Teacher3': net3.state_dict() if use_cuda else net3,
                'Teacher4': net4.state_dict() if use_cuda else net4,
                'Teacher5': net5.state_dict() if use_cuda else net5,
                'Teacher6': net6.state_dict() if use_cuda else net6,
                'Teacher7': net7.state_dict() if use_cuda else net7,
                'Teacher8': net8.state_dict() if use_cuda else net8,
                'test_Teacher1_accuracy': test_acc1,
                'test_Teacher2_accuracy': test_acc2,
                'test_Teacher3_accuracy': test_acc3,
                'test_Teacher4_accuracy': test_acc4,
                'test_Teacher5_accuracy': test_acc5,
                'test_Teacher6_accuracy': test_acc6,
                'test_Teacher7_accuracy': test_acc7,
                'test_Teacher8_accuracy': test_acc8,
                'test_Avg_accuracy': test_avgACC, 
                'test_Avg_MAP': test_avgMAP,
                'test_Avg_F1': test_avgF1, 
                'test_epoch': epoch,
            } 
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(state, os.path.join(path,'Best_MultiTeacher_model.t7'))
    
    else:
        raise Exception('Invalid ...')

print("best_PrivateTest_avgACC: %0.3f" % best_ACC)
writer.close()

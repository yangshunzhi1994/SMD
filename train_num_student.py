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
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from datasets.RAF import RAF
from datasets.PET import PET
from datasets.colorferet import colorferet
from datasets.FairFace import FairFace
from network.teacherNet import Teacher
from network.studentNet import CNN_RIS
import other

import utils
from utils import load_pretrained_model, count_parameters_in_MB

import losses
from tensorboardX import SummaryWriter
from network.num_student import Dynamic_MultiTeacher2,Dynamic_MultiTeacher3,Dynamic_MultiTeacher4,\
               Dynamic_MultiTeacher5,Dynamic_MultiTeacher6,Dynamic_MultiTeacher7, Dynamic_MultiTeacher8

parser = argparse.ArgumentParser(description='train kd')

# various path
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--t_model', type=str, default="Teacher", help='Teacher,Teacher1,Teacher3')
parser.add_argument('--s_model', type=str, default="CNNRIS", help='name of student model')
parser.add_argument('--fusion', type=str, default="OurDiversity", help='OurDiversity')
parser.add_argument('--distillation', type=str, default="KD", help='KD')
parser.add_argument('--data_name', type=str, default='RAF', help='RAF,FairFace,colorferet,PET') 

# training hyper parameters
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='learning rate')
parser.add_argument('--test_bs', default=256, type=int, help='learning rate')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')#1e-4,5e-4
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
parser.add_argument('--noise', type=str, default='None', help='GaussianBlur,AverageBlur,MedianBlur,BilateralFilter,Salt-and-pepper')
parser.add_argument('--number_teacher', default=2, type=int, help='Batch size')

args, unparsed = parser.parse_known_args()

path = os.path.join(args.save_root + args.data_name+ '_MultiTeacher_OurDiversity_NumberTeacher_' + str(args.number_teacher)+ '_' + args.distillation)
writer = SummaryWriter(log_dir=path)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
else:
    pass

tcheckpoint = torch.load(os.path.join('results/' + args.data_name+ '_MultiTeacher_OurDiversity_NumberTeacher_'+ \
                                      str(args.number_teacher),'Best_MultiTeacher_model.t7'))

print ('The dataset used for training is:   '+ str(args.data_name))
print ('The number of teachers is :       '+ str(args.number_teacher))

if args.number_teacher == 2:
    tnet1 = Teacher().cuda()
    tnet2 = Teacher().cuda()
    load_pretrained_model(tnet1, tcheckpoint['Teacher1'])
    load_pretrained_model(tnet2, tcheckpoint['Teacher2'])
    print ('best_Teacher1_acc is '+ str(tcheckpoint['test_Teacher1_accuracy']))
    print ('best_Teacher2_acc is '+ str(tcheckpoint['test_Teacher2_accuracy'])) 
    tnet1.eval()
    for param in tnet1.parameters():
        param.requires_grad = False
    tnet2.eval()
    for param in tnet2.parameters():
        param.requires_grad = False
elif args.number_teacher == 3:
    tnet1 = Teacher().cuda()
    tnet2 = Teacher().cuda()
    tnet3 = Teacher().cuda()
    load_pretrained_model(tnet1, tcheckpoint['Teacher1'])
    load_pretrained_model(tnet2, tcheckpoint['Teacher2'])
    load_pretrained_model(tnet3, tcheckpoint['Teacher3'])
    print ('best_Teacher1_acc is '+ str(tcheckpoint['test_Teacher1_accuracy']))
    print ('best_Teacher2_acc is '+ str(tcheckpoint['test_Teacher2_accuracy'])) 
    print ('best_Teacher3_acc is '+ str(tcheckpoint['test_Teacher3_accuracy'])) 
    tnet1.eval()
    for param in tnet1.parameters():
        param.requires_grad = False
    tnet2.eval()
    for param in tnet2.parameters():
        param.requires_grad = False
    tnet3.eval()
    for param in tnet3.parameters():
        param.requires_grad = False
elif args.number_teacher == 4:
    tnet1 = Teacher().cuda()
    tnet2 = Teacher().cuda()
    tnet3 = Teacher().cuda()
    tnet4 = Teacher().cuda()
    load_pretrained_model(tnet1, tcheckpoint['Teacher1'])
    load_pretrained_model(tnet2, tcheckpoint['Teacher2'])
    load_pretrained_model(tnet3, tcheckpoint['Teacher3'])
    load_pretrained_model(tnet4, tcheckpoint['Teacher4'])
    print ('best_Teacher1_acc is '+ str(tcheckpoint['test_Teacher1_accuracy']))
    print ('best_Teacher2_acc is '+ str(tcheckpoint['test_Teacher2_accuracy'])) 
    print ('best_Teacher3_acc is '+ str(tcheckpoint['test_Teacher3_accuracy'])) 
    print ('best_Teacher4_acc is '+ str(tcheckpoint['test_Teacher4_accuracy']))
    tnet1.eval()
    for param in tnet1.parameters():
        param.requires_grad = False
    tnet2.eval()
    for param in tnet2.parameters():
        param.requires_grad = False
    tnet3.eval()
    for param in tnet3.parameters():
        param.requires_grad = False
    tnet4.eval()
    for param in tnet4.parameters():
        param.requires_grad = False
elif args.number_teacher == 5:
    tnet1 = Teacher().cuda()
    tnet2 = Teacher().cuda()
    tnet3 = Teacher().cuda()
    tnet4 = Teacher().cuda()
    tnet5 = Teacher().cuda()
    load_pretrained_model(tnet1, tcheckpoint['Teacher1'])
    load_pretrained_model(tnet2, tcheckpoint['Teacher2'])
    load_pretrained_model(tnet3, tcheckpoint['Teacher3'])
    load_pretrained_model(tnet4, tcheckpoint['Teacher4'])
    load_pretrained_model(tnet5, tcheckpoint['Teacher5'])
    print ('best_Teacher1_acc is '+ str(tcheckpoint['test_Teacher1_accuracy']))
    print ('best_Teacher2_acc is '+ str(tcheckpoint['test_Teacher2_accuracy'])) 
    print ('best_Teacher3_acc is '+ str(tcheckpoint['test_Teacher3_accuracy'])) 
    print ('best_Teacher4_acc is '+ str(tcheckpoint['test_Teacher4_accuracy']))
    print ('best_Teacher5_acc is '+ str(tcheckpoint['test_Teacher5_accuracy']))
    tnet1.eval()
    for param in tnet1.parameters():
        param.requires_grad = False
    tnet2.eval()
    for param in tnet2.parameters():
        param.requires_grad = False
    tnet3.eval()
    for param in tnet3.parameters():
        param.requires_grad = False
    tnet4.eval()
    for param in tnet4.parameters():
        param.requires_grad = False
    tnet5.eval()
    for param in tnet5.parameters():
        param.requires_grad = False
elif args.number_teacher == 6:
    tnet1 = Teacher().cuda()
    tnet2 = Teacher().cuda()
    tnet3 = Teacher().cuda()
    tnet4 = Teacher().cuda()
    tnet5 = Teacher().cuda()
    tnet6 = Teacher().cuda()
    load_pretrained_model(tnet1, tcheckpoint['Teacher1'])
    load_pretrained_model(tnet2, tcheckpoint['Teacher2'])
    load_pretrained_model(tnet3, tcheckpoint['Teacher3'])
    load_pretrained_model(tnet4, tcheckpoint['Teacher4'])
    load_pretrained_model(tnet5, tcheckpoint['Teacher5'])
    load_pretrained_model(tnet6, tcheckpoint['Teacher6'])
    print ('best_Teacher1_acc is '+ str(tcheckpoint['test_Teacher1_accuracy']))
    print ('best_Teacher2_acc is '+ str(tcheckpoint['test_Teacher2_accuracy'])) 
    print ('best_Teacher3_acc is '+ str(tcheckpoint['test_Teacher3_accuracy'])) 
    print ('best_Teacher4_acc is '+ str(tcheckpoint['test_Teacher4_accuracy']))
    print ('best_Teacher5_acc is '+ str(tcheckpoint['test_Teacher5_accuracy']))
    print ('best_Teacher6_acc is '+ str(tcheckpoint['test_Teacher6_accuracy']))
    tnet1.eval()
    for param in tnet1.parameters():
        param.requires_grad = False
    tnet2.eval()
    for param in tnet2.parameters():
        param.requires_grad = False
    tnet3.eval()
    for param in tnet3.parameters():
        param.requires_grad = False
    tnet4.eval()
    for param in tnet4.parameters():
        param.requires_grad = False   
    tnet5.eval()
    for param in tnet5.parameters():
        param.requires_grad = False 
    tnet6.eval()
    for param in tnet6.parameters():
        param.requires_grad = False 
elif args.number_teacher == 7:
    tnet1 = Teacher().cuda()
    tnet2 = Teacher().cuda()
    tnet3 = Teacher().cuda()
    tnet4 = Teacher().cuda()
    tnet5 = Teacher().cuda()
    tnet6 = Teacher().cuda()
    tnet7 = Teacher().cuda()
    load_pretrained_model(tnet1, tcheckpoint['Teacher1'])
    load_pretrained_model(tnet2, tcheckpoint['Teacher2'])
    load_pretrained_model(tnet3, tcheckpoint['Teacher3'])
    load_pretrained_model(tnet4, tcheckpoint['Teacher4'])
    load_pretrained_model(tnet5, tcheckpoint['Teacher5'])
    load_pretrained_model(tnet6, tcheckpoint['Teacher6'])
    load_pretrained_model(tnet7, tcheckpoint['Teacher7'])
    print ('best_Teacher1_acc is '+ str(tcheckpoint['test_Teacher1_accuracy']))
    print ('best_Teacher2_acc is '+ str(tcheckpoint['test_Teacher2_accuracy'])) 
    print ('best_Teacher3_acc is '+ str(tcheckpoint['test_Teacher3_accuracy'])) 
    print ('best_Teacher4_acc is '+ str(tcheckpoint['test_Teacher4_accuracy']))
    print ('best_Teacher5_acc is '+ str(tcheckpoint['test_Teacher5_accuracy']))
    print ('best_Teacher6_acc is '+ str(tcheckpoint['test_Teacher6_accuracy']))
    print ('best_Teacher7_acc is '+ str(tcheckpoint['test_Teacher7_accuracy']))
    tnet1.eval()
    for param in tnet1.parameters():
        param.requires_grad = False
    tnet2.eval()
    for param in tnet2.parameters():
        param.requires_grad = False
    tnet3.eval()
    for param in tnet3.parameters():
        param.requires_grad = False
    tnet4.eval()
    for param in tnet4.parameters():
        param.requires_grad = False   
    tnet5.eval()
    for param in tnet5.parameters():
        param.requires_grad = False 
    tnet6.eval()
    for param in tnet6.parameters():
        param.requires_grad = False 
    tnet7.eval()
    for param in tnet7.parameters():
        param.requires_grad = False 
elif args.number_teacher == 8:
    tnet1 = Teacher().cuda()
    tnet2 = Teacher().cuda()
    tnet3 = Teacher().cuda()
    tnet4 = Teacher().cuda()
    tnet5 = Teacher().cuda()
    tnet6 = Teacher().cuda()
    tnet7 = Teacher().cuda()
    tnet8 = Teacher().cuda()
    load_pretrained_model(tnet1, tcheckpoint['Teacher1'])
    load_pretrained_model(tnet2, tcheckpoint['Teacher2'])
    load_pretrained_model(tnet3, tcheckpoint['Teacher3'])
    load_pretrained_model(tnet4, tcheckpoint['Teacher4'])
    load_pretrained_model(tnet5, tcheckpoint['Teacher5'])
    load_pretrained_model(tnet6, tcheckpoint['Teacher6'])
    load_pretrained_model(tnet7, tcheckpoint['Teacher7'])
    load_pretrained_model(tnet8, tcheckpoint['Teacher8'])
    print ('best_Teacher1_acc is '+ str(tcheckpoint['test_Teacher1_accuracy']))
    print ('best_Teacher2_acc is '+ str(tcheckpoint['test_Teacher2_accuracy'])) 
    print ('best_Teacher3_acc is '+ str(tcheckpoint['test_Teacher3_accuracy'])) 
    print ('best_Teacher4_acc is '+ str(tcheckpoint['test_Teacher4_accuracy']))
    print ('best_Teacher5_acc is '+ str(tcheckpoint['test_Teacher5_accuracy']))
    print ('best_Teacher6_acc is '+ str(tcheckpoint['test_Teacher6_accuracy']))
    print ('best_Teacher7_acc is '+ str(tcheckpoint['test_Teacher7_accuracy']))
    print ('best_Teacher8_acc is '+ str(tcheckpoint['test_Teacher8_accuracy']))
    tnet1.eval()
    for param in tnet1.parameters():
        param.requires_grad = False
    tnet2.eval()
    for param in tnet2.parameters():
        param.requires_grad = False
    tnet3.eval()
    for param in tnet3.parameters():
        param.requires_grad = False
    tnet4.eval()
    for param in tnet4.parameters():
        param.requires_grad = False   
    tnet5.eval()
    for param in tnet5.parameters():
        param.requires_grad = False 
    tnet6.eval()
    for param in tnet6.parameters():
        param.requires_grad = False 
    tnet7.eval()
    for param in tnet7.parameters():
        param.requires_grad = False 
    tnet8.eval()
    for param in tnet8.parameters():
        param.requires_grad = False 
else:
    raise Exception('Invalid name of the teacher network...')

print ('best_Teacher_Avg_accuracy is '+ str(tcheckpoint['test_Avg_accuracy'])) 
print ('best_Teacher_Avg_MAP is '+ str(tcheckpoint['test_Avg_MAP'])) 
print ('best_Teacher_Avg_F1 is '+ str(tcheckpoint['test_Avg_F1'])) 

snet = CNN_RIS().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(snet.parameters(), lr = args.lr, momentum = args.momentum,
                            weight_decay = args.weight_decay,nesterov = True)

transform_train = transforms.Compose([
    transforms.RandomCrop(92),
    transforms.RandomHorizontalFlip(),
])

if args.data_name == 'RAF':
	transforms_teacher_Normalize = transforms.Normalize((0.5884594, 0.45767313, 0.40865755), 
                            (0.25717735, 0.23602168, 0.23505741))
	transforms_student_Normalize =  transforms.Normalize((0.58846486, 0.45766878, 0.40865615), 
                            (0.2516557, 0.23020789, 0.22939532))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.59003043, 0.4573948, 0.40749523], std=[0.2465465, 0.22635746, 0.22564183])
            (transforms.ToTensor()(crop)) for crop in crops]))
elif args.data_name == 'PET':
	transforms_teacher_Normalize = transforms.Normalize((0.47950855, 0.4454716, 0.3953508), 
                            (0.26221144, 0.25676072, 0.2640482))
	transforms_student_Normalize =  transforms.Normalize((0.4794851, 0.44543326, 0.39531776), 
                            (0.24786888, 0.24236518, 0.24950708))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.4862494, 0.45275217, 0.39576027], std=[0.24864933, 0.2446337, 0.2527274])
            (transforms.ToTensor()(crop)) for crop in crops]))
elif args.data_name == 'colorferet':
	transforms_teacher_Normalize = transforms.Normalize((0.50150657, 0.4387828, 0.37715995), 
                            (0.22249317, 0.24526535, 0.25831717))
	transforms_student_Normalize =  transforms.Normalize((0.50166893, 0.43892872, 0.37727863), 
                            (0.21588857, 0.23875234, 0.25212118))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.4992823, 0.4371743, 0.37574747], std=[0.21377444, 0.23534843, 0.24466512])
            (transforms.ToTensor()(crop)) for crop in crops]))
elif args.data_name == 'FairFace':
	transforms_teacher_Normalize = transforms.Normalize((0.4911152, 0.36028033, 0.30489963), 
                            (0.25160596, 0.21829675, 0.21198231))
	transforms_student_Normalize =  transforms.Normalize((0.4911364, 0.3602937, 0.3049148), 
                            (0.24722975, 0.21383813, 0.20771481))
	transforms_test_Normalize = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(
            mean=[0.49202734, 0.36110377, 0.30535242], std=[0.24179104, 0.21022305, 0.20413795])
            (transforms.ToTensor()(crop)) for crop in crops]))
else:
    raise Exception('Invalid dataset name...')

teacher_norm = transforms.Compose([
transforms.ToTensor(),
transforms_teacher_Normalize,
])

student_norm = transforms.Compose([
transforms.Resize(44),
transforms.ToTensor(),
transforms_student_Normalize,
])

transform_test = transforms.Compose([
transforms.TenCrop(44),
transforms_test_Normalize,
])

if args.data_name == 'RAF':
	trainset = RAF(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm, noise=args.noise)
	PrivateTestset = RAF(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None, noise=args.noise)
elif args.data_name == 'PET':
	trainset = PET(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm, noise=args.noise)
	PrivateTestset = PET(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None, noise=args.noise)
elif args.data_name == 'colorferet':
	trainset = colorferet(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm, noise=args.noise)
	PrivateTestset = colorferet(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None, noise=args.noise)
elif args.data_name == 'FairFace':
	trainset = FairFace(split = 'Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm, noise=args.noise)
	PrivateTestset = FairFace(split = 'PrivateTest', transform=transform_test, student_norm=None, teacher_norm=None, noise=args.noise)
else:
	raise Exception('Invalid dataset name...')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=args.test_bs, shuffle=False, num_workers=args.num_workers)

best_acc = 0
best_mAP = 0
best_F1 = 0
NUM_CLASSES = 7
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

def train(epoch):
    print('\nEpoch: %d' % epoch)
    snet.train()
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

    for batch_idx, (img_teacher, img_student, target) in enumerate(trainloader):
        if args.cuda:
            img_teacher = img_teacher.cuda()
            img_student = img_student.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        img_teacher, img_student, target = Variable(img_teacher), Variable(img_student), Variable(target)
        
        rb1_s, rb2_s, rb3_s, mimic_s, out_s = snet(img_student)
        with torch.no_grad():
            if args.number_teacher == 2:
                _, _, _, _, out_t1 = tnet1(img_teacher)
                _, _, _, _, out_t2 = tnet2(img_teacher)
            elif args.number_teacher == 3:
                _, _, _, _, out_t1 = tnet1(img_teacher)
                _, _, _, _, out_t2 = tnet2(img_teacher)
                _, _, _, _, out_t3 = tnet3(img_teacher)
            elif args.number_teacher == 4:
                _, _, _, _, out_t1 = tnet1(img_teacher)
                _, _, _, _, out_t2 = tnet2(img_teacher)
                _, _, _, _, out_t3 = tnet3(img_teacher)
                _, _, _, _, out_t4 = tnet4(img_teacher)
            elif args.number_teacher == 5:
                _, _, _, _, out_t1 = tnet1(img_teacher)
                _, _, _, _, out_t2 = tnet2(img_teacher)
                _, _, _, _, out_t3 = tnet3(img_teacher)
                _, _, _, _, out_t4 = tnet4(img_teacher)
                _, _, _, _, out_t5 = tnet5(img_teacher)
            elif args.number_teacher == 6:
                _, _, _, _, out_t1 = tnet1(img_teacher)
                _, _, _, _, out_t2 = tnet2(img_teacher)
                _, _, _, _, out_t3 = tnet3(img_teacher)
                _, _, _, _, out_t4 = tnet4(img_teacher)
                _, _, _, _, out_t5 = tnet5(img_teacher)
                _, _, _, _, out_t6 = tnet6(img_teacher)
            elif args.number_teacher == 7:
                _, _, _, _, out_t1 = tnet1(img_teacher)
                _, _, _, _, out_t2 = tnet2(img_teacher)
                _, _, _, _, out_t3 = tnet3(img_teacher)
                _, _, _, _, out_t4 = tnet4(img_teacher)
                _, _, _, _, out_t5 = tnet5(img_teacher)
                _, _, _, _, out_t6 = tnet6(img_teacher)
                _, _, _, _, out_t7 = tnet7(img_teacher)
            elif args.number_teacher == 8:
                _, _, _, _, out_t1 = tnet1(img_teacher)
                _, _, _, _, out_t2 = tnet2(img_teacher)
                _, _, _, _, out_t3 = tnet3(img_teacher)
                _, _, _, _, out_t4 = tnet4(img_teacher)
                _, _, _, _, out_t5 = tnet5(img_teacher)
                _, _, _, _, out_t6 = tnet6(img_teacher)
                _, _, _, _, out_t7 = tnet7(img_teacher)
                _, _, _, _, out_t8 = tnet8(img_teacher)
            else:
                raise Exception('Invalid ...')
        if args.number_teacher == 2:
            loss = Dynamic_MultiTeacher2().cuda()(out_t1, out_t2, out_s, target)
        elif args.number_teacher == 3:
            loss = Dynamic_MultiTeacher3().cuda()(out_t1, out_t2, out_t3, out_s, target)
        elif args.number_teacher == 4:
            loss = Dynamic_MultiTeacher4().cuda()(out_t1, out_t2, out_t3, out_t4, out_s, target)
        elif args.number_teacher == 5:
            loss = Dynamic_MultiTeacher5().cuda()(out_t1, out_t2, out_t3, out_t4, out_t5, out_s, target)
        elif args.number_teacher == 6:
            loss = Dynamic_MultiTeacher6().cuda()(out_t1, out_t2, out_t3, out_t4, out_t5, out_t6, out_s, target)
        elif args.number_teacher == 7:
            loss = Dynamic_MultiTeacher7().cuda()(out_t1, out_t2, out_t3, out_t4, out_t5, out_t6, out_t7, out_s, target)
        elif args.number_teacher == 8:
            loss = Dynamic_MultiTeacher8().cuda()(out_t1, out_t2, out_t3, out_t4, out_t5, out_t6, out_t7, out_t8, out_s, target)
        else:
            raise Exception('Invalid ...')
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        
        conf_mat += losses.confusion_matrix(out_s, target, NUM_CLASSES)
        acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
        precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
        mAP = sum(precision)/len(precision)
        
        recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
        precision = np.array(precision)
        recall = np.array(recall)
        f1 = 2 * precision*recall/(precision+recall + 1e-10)
        F1_score = f1.mean()
    
    return train_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score

def test(epoch):
    snet.eval()
    PrivateTest_loss = 0
    t_prediction = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for batch_idx, (img, target) in enumerate(PrivateTestloader):
        t = time.time()
        test_bs, ncrops, c, h, w = np.shape(img)
        img = img.view(-1, c, h, w)
        if args.cuda:
            img = img.cuda()
            target = target.cuda()
        img, target = Variable(img), Variable(target)
        with torch.no_grad():
            rb1_s, rb2_s, rb3_s, mimic_s, out_s = snet(img)
            
        outputs_avg = out_s.view(test_bs, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, target)
        t_prediction += (time.time() - t)
        PrivateTest_loss += loss.item()
        
        conf_mat += losses.confusion_matrix(outputs_avg, target, NUM_CLASSES)
        acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
        precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
        mAP = sum(precision)/len(precision)
        
        recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
        precision = np.array(precision)
        recall = np.array(recall)
        f1 = 2 * precision*recall/(precision+recall + 1e-10)
        F1_score = f1.mean()
  
    return PrivateTest_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score

for epoch in range(1, args.epochs+1):
    train_loss, train_acc, train_mAP, train_F1 = train(epoch)
    test_loss, test_acc, test_mAP, test_F1 = test(epoch)
    print("train_loss:  %0.3f, train_acc:  %0.3f, train_mAP:  %0.3f, train_F1:  %0.3f"%
          (train_loss, train_acc, train_mAP, train_F1))
    print("test_loss:   %0.3f, test_acc:   %0.3f, test_mAP:   %0.3f, test_F1:   %0.3f"%
          (test_loss, test_acc, test_mAP, test_F1))
    writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
    writer.add_scalars('epoch/accuracy', {'train': train_acc, 'test': test_acc}, epoch)
    writer.add_scalars('epoch/mAP', {'train': train_mAP, 'test': test_mAP}, epoch)
    writer.add_scalars('epoch/F1', {'train': train_F1, 'test': test_F1}, epoch)
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_mAP = test_mAP
        best_F1 = test_F1
        print ('Saving models......')
        print("best_PrivateTest_acc: %0.3f" % best_acc)
        print("best_PrivateTest_mAP: %0.3f" % best_mAP)
        print("best_PrivateTest_F1: %0.3f" % best_F1)
        state = {
            'epoch': epoch,
            'snet': snet.state_dict() if args.cuda else snet,
            'test_acc': test_acc,
            'test_mAP': test_mAP,
            'test_F1': test_F1,
            'test_epoch': epoch,
        } 
        torch.save(state, os.path.join(path,'Student_Test_model.t7'))

print("best_PrivateTest_acc: %0.3f" % best_acc)
print("best_PrivateTest_mAP: %0.3f" % best_mAP)
print("best_PrivateTest_F1: %0.3f" % best_F1)




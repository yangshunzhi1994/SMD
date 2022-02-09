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
parser.add_argument('--data_name', type=str, default="FairFace", help='RAF,FairFace,colorferet,PET')
parser.add_argument('--epochs', type=int, default=400, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='Batch size')
parser.add_argument('--test_bs', default=256, type=int, help='Batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--direction', default=0, type=float, help='direction')
parser.add_argument('--variance', default=0, type=float, help='variance')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
parser.add_argument('--fusion', type=str, default="OurDiversity", 
                    help='OurDiversity,Average,AEKD,EDM,RIDE,KDCL,AED,ENDD,Few-Shot,EKD')
                    # OurDiversity,Average,AEKD,EDM,RIDE,KDCL,AED,ENDD,Few-Shot,FFMCD,OKDDip,ONE,PCL
args = parser.parse_args()

best_ACC = 0
learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9
if args.data_name == 'colorferet':
    NUM_CLASSES = 994
elif args.data_name == 'PET':
    NUM_CLASSES = 37
else:
    NUM_CLASSES = 7

total_epoch = args.epochs

if args.fusion == 'OurDiversity':
    path = os.path.join(args.save_root + args.data_name + '_' + args.model+ '_' + args.fusion+ '_' + str(args.direction)+ '_' + str(args.variance))
else:
    path = os.path.join(args.save_root + args.data_name + '_' + args.model+ '_' + args.fusion)
writer = SummaryWriter(log_dir=path)

# Data
print ('The dataset used for training is:                '+ str(args.data_name))
print ('The training mode is:                        '+ str(args.model))
print ('The fusion method is:                        '+ str(args.fusion))
print ('The direction is:                          '+ str(args.direction))
print ('The variance is:                           '+ str(args.variance))
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

net1 = Teacher(num_classes=NUM_CLASSES).cuda()
net2 = Teacher(num_classes=NUM_CLASSES).cuda()
net3 = Teacher(num_classes=NUM_CLASSES).cuda()
net4 = Teacher(num_classes=NUM_CLASSES).cuda()
criterion = nn.CrossEntropyLoss().cuda()

if args.fusion == 'OKDDip' or args.fusion == 'PCL': 
    if args.fusion == 'OKDDip':
        net = other.OKDDip(in_dim=NUM_CLASSES, out_dim=NUM_CLASSES)
    else:
        net = other.PCL(out_dim=NUM_CLASSES)
    net.cuda()
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters(),net3.parameters(),net4.parameters(),net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = optim.SGD(itertools.chain(net1.parameters(),net2.parameters(),net3.parameters(),net4.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net1.train()
    net2.train()
    net3.train()
    net4.train()
    if args.fusion == 'OKDDip' or args.fusion == 'PCL': 
        net.train()
    else:
        pass
    train_loss = 0

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = args.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = args.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        inputs, targets = Variable(inputs), Variable(targets)
        
        _, _, _, mimic1, outputs1 = net1(inputs)
        _, _, _, mimic2, outputs2 = net2(inputs)
        _, _, _, mimic3, outputs3 = net3(inputs)
        _, _, _, mimic4, outputs4 = net4(inputs)
            
        loss1 = criterion(outputs1, targets)
        loss2 = criterion(outputs2, targets)
        loss3 = criterion(outputs3, targets)
        loss4 = criterion(outputs4, targets)
        
        if args.fusion == 'Average':
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2 + loss3 + loss4 
        elif args.fusion == 'AEKD':
            mimic = other.AEKD_Teacher().cuda()(outputs1,outputs2,outputs3,outputs4)
            loss = criterion(mimic, targets)
            loss = loss + loss1 + loss2 + loss3 + loss4 
        elif args.fusion == 'OurDiversity':
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            loss = criterion(mimic, targets)
            diversityLoss = losses.Diversity(args.direction, args.variance).cuda()(outputs1,outputs2,outputs3,outputs4,targets)
            loss = loss + loss1 + loss2 + loss3 + loss4 + diversityLoss
        elif args.fusion == 'EDM':# Protecting DNNs from Theft using an Ensemble of Diverse Models 
            emd_loss = other.EDM().cuda()(outputs1,outputs2,outputs3,outputs4)
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            loss = 0.25*loss1 + 0.25*loss2 + 0.25*loss3 + 0.25*loss4 + emd_loss
        elif args.fusion == 'RIDE':# RIDE: Long-tailed Recognition by Routing Diverse Distribution-Aware Experts.
            ride_loss = other.RIDE().cuda()(outputs1,outputs2,outputs3,outputs4,targets)
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            loss = loss1 + loss2 + loss3 + loss4 - 0.2*ride_loss
        elif args.fusion == 'KDCL':# Online Knowledge Distillation via Collaborative Learning
            kdcl_loss, mimic = other.KDCL().cuda()(outputs1,outputs2,outputs3,outputs4,targets)
            loss = loss1 + loss2 + loss3 + loss4 + kdcl_loss
        elif args.fusion == 'AED':# Adaptable Ensemble Distillation
            loss, mimic = other.AED().cuda()(outputs1,outputs2,outputs3,outputs4,targets)
        elif args.fusion == 'ENDD':# Ensemble Distribution Distillation
            outputs1 = torch.exp(outputs1)
            outputs2 = torch.exp(outputs2)
            outputs3 = torch.exp(outputs3)
            outputs4 = torch.exp(outputs4)
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            loss = criterion(mimic, targets)
            loss = loss1 + loss2 + loss3 + loss4 + loss
        elif args.fusion == 'Few-Shot':# Diversity with Cooperation: Ensemble Methods for Few-Shot Classification
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            loss = other.FewShot().cuda()(outputs1,outputs2,outputs3,outputs4,targets)
            loss = loss1 + loss2 + loss3 + loss4 + loss
        elif args.fusion == 'FFMCD':#Online Knowledge Distillation via Multi-branch Diversity Enhancement
            loss_CD = other.CD().cuda()(net1,net2,net3,net4)
            loss = other.FFMCD().cuda()(outputs1,outputs2,outputs3,outputs4,targets)
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            loss = loss + 0.00000005*loss_CD
        elif args.fusion == 'OKDDip':#Online Knowledge Distillation with Diverse Peers
            L_dis = net(outputs1,outputs2,outputs3,outputs4)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            loss4 = criterion(outputs4, targets)
            loss = loss1 + loss2 + loss3 + loss4 + L_dis
            mimic = outputs4
        elif args.fusion == 'ONE':# Knowledge Distillation by On-the-Fly Native Ensemble
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            loss4 = criterion(outputs4, targets)
            
            mimic = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), outputs4.unsqueeze(0)),0)
            g = F.softmax(mimic,dim=0)
            mimic = g[0]*outputs1 + g[1]*outputs2 + g[2]*outputs3 + g[3]*outputs4
            mimic_loss = criterion(mimic, targets)
            
            kl_loss1 = other.KL_divergence(temperature = 3).cuda()(mimic,outputs1)
            kl_loss2 = other.KL_divergence(temperature = 3).cuda()(mimic,outputs2)
            kl_loss3 = other.KL_divergence(temperature = 3).cuda()(mimic,outputs3)
            kl_loss4 = other.KL_divergence(temperature = 3).cuda()(mimic,outputs4)
            loss = loss1+loss2+loss3+loss4+mimic_loss+kl_loss1+kl_loss2+kl_loss3+kl_loss4
        elif args.fusion == 'PCL':# Peer Collaborative Learning for Online Knowledge Distillation
            mimic = net(mimic1,mimic2,mimic3,mimic4)
            loss1 = criterion(outputs1, targets)
            loss2 = criterion(outputs2, targets)
            loss3 = criterion(outputs3, targets)
            loss4 = criterion(outputs4, targets)
            mimic_loss = criterion(mimic, targets)
            kl_loss1 = other.sigmoid_rampup(epoch, 80)*other.KL_divergence(temperature = 3).cuda()(mimic,outputs1)
            kl_loss2 = other.sigmoid_rampup(epoch, 80)*other.KL_divergence(temperature = 3).cuda()(mimic,outputs2)
            kl_loss3 = other.sigmoid_rampup(epoch, 80)*other.KL_divergence(temperature = 3).cuda()(mimic,outputs3)
            kl_loss4 = other.sigmoid_rampup(epoch, 80)*other.KL_divergence(temperature = 3).cuda()(mimic,outputs4)
            loss = loss1+loss2+loss3+loss4+mimic_loss+kl_loss1+kl_loss2+kl_loss3+kl_loss4
        elif args.fusion == 'EKD':#Ensemble Knowledge Distillation for Learning Improved and Efficient Networks
            mimic = outputs1+outputs2+outputs3+outputs4
            tea_cls_loss = criterion(mimic, targets)
            EKD_loss = other.EKD_Teacher().cuda()(outputs1,outputs2,outputs3,outputs4,mimic)
            loss = 0.5*tea_cls_loss+0.6*EKD_loss
        else:
            raise Exception('Invalid ...')
        
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        
        train_loss += loss.item()
        
        conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, mimic, targets, NUM_CLASSES)
        conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
        conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
        conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
        conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
    
    return train_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc, 100.* mAP, 100 * F1_score

def test(epoch):
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()
    
    if args.fusion == 'OKDDip' or args.fusion == 'PCL': 
        net.eval()
    else:
        pass
    
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat1 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat2 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat3 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat4 = np.zeros((NUM_CLASSES, NUM_CLASSES))
    
    PrivateTest_loss = 0

    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        t = time.time()
        test_bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        
        with torch.no_grad():
            _, _, _, mimic1, outputs1 = net1(inputs)
            _, _, _, mimic2, outputs2 = net2(inputs)
            _, _, _, mimic3, outputs3 = net3(inputs)
            _, _, _, mimic4, outputs4 = net4(inputs)
            
        outputs1 = outputs1.view(test_bs, ncrops, -1).mean(1)
        outputs2 = outputs2.view(test_bs, ncrops, -1).mean(1) 
        outputs3 = outputs3.view(test_bs, ncrops, -1).mean(1) 
        outputs4 = outputs4.view(test_bs, ncrops, -1).mean(1)
        
        if args.fusion == 'AEKD':
            mimic = other.AEKD_Teacher().cuda()(outputs1,outputs2,outputs3,outputs4)
        elif args.fusion == 'FFMCD':
            mimic = torch.cat((outputs1.unsqueeze(0),outputs2.unsqueeze(0),outputs3.unsqueeze(0),outputs4.unsqueeze(0)),0)
            mimic = F.softmax(mimic,dim=2)
            mimic = mimic[0]*outputs1+mimic[1]*outputs2+mimic[2]*outputs3+mimic[3]*outputs4
        elif args.fusion == 'ENDD':
            outputs1 = torch.exp(outputs1)
            outputs2 = torch.exp(outputs2)
            outputs3 = torch.exp(outputs3)
            outputs4 = torch.exp(outputs4)
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
        elif args.fusion == 'ONE':# Knowledge Distillation by On-the-Fly Native Ensemble
            mimic = torch.cat((outputs1.unsqueeze(0), outputs2.unsqueeze(0), outputs3.unsqueeze(0), outputs4.unsqueeze(0)),0)
            g = F.softmax(mimic,dim=0)
            mimic = g[0]*outputs1 + g[1]*outputs2 + g[2]*outputs3 + g[3]*outputs4
        elif args.fusion == 'PCL':# Peer Collaborative Learning for Online Knowledge Distillation
            mimic1 = mimic1.view(test_bs, ncrops, -1).mean(1)
            mimic2 = mimic2.view(test_bs, ncrops, -1).mean(1)
            mimic3 = mimic3.view(test_bs, ncrops, -1).mean(1) 
            mimic4 = mimic4.view(test_bs, ncrops, -1).mean(1)
            mimic = net(mimic1,mimic2,mimic3,mimic4)
        else:
            mimic = (outputs1+outputs2+outputs3+outputs4)/4
            
        loss = criterion(mimic, targets)
        PrivateTest_loss += loss.item()
        
        conf_mat, acc, mAP, F1_score = ACC_evaluation(conf_mat, mimic, targets, NUM_CLASSES)
        conf_mat1, acc1, mAP1, F1_score1 = ACC_evaluation(conf_mat1, outputs1, targets, NUM_CLASSES)
        conf_mat2, acc2, mAP2, F1_score2 = ACC_evaluation(conf_mat2, outputs2, targets, NUM_CLASSES)
        conf_mat3, acc3, mAP3, F1_score3 = ACC_evaluation(conf_mat3, outputs3, targets, NUM_CLASSES)
        conf_mat4, acc4, mAP4, F1_score4 = ACC_evaluation(conf_mat4, outputs4, targets, NUM_CLASSES)
    
    return PrivateTest_loss/(batch_idx+1), 100.*acc1, 100.*acc2, 100.*acc3, 100.*acc4, 100.*acc, 100.* mAP, 100 * F1_score


for epoch in range(0, total_epoch):
    train_loss, train_acc1, train_acc2, train_acc3, train_acc4,train_avgACC, train_avgMAP, train_avgF1 = train(epoch)
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
            'Teacher1': net1.state_dict(),
            'Teacher2': net2.state_dict(),
            'Teacher3': net3.state_dict(),
            'Teacher4': net4.state_dict(),
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

print("best_PrivateTest_avgACC: %0.3f" % best_ACC)
writer.close()

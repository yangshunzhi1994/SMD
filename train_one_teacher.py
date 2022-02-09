from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import time
import argparse
import utils
import losses
from utils import load_pretrained_model
from datasets.RAF import RAF_multi_teacher
from datasets.PET import PET_multi_teacher
from datasets.colorferet import colorferet_multi_teacher
from datasets.FairFace import FairFace_multi_teacher
from torch.autograd import Variable
from network.teacherNet import Teacher
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Teacher CNN Training')
parser.add_argument('--save_root', type=str, default='results/', help='models and logs are saved here')
parser.add_argument('--model', type=str, default="Teacher", help='Teacher')
parser.add_argument('--data_name', type=str, default="RAF", help='RAF,FairFace,colorferet,PET')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--train_bs', default=32, type=int, help='Batch size')
parser.add_argument('--test_bs', default=8, type=int, help='Batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--augmentation', default=False, type=int, help='use mixup and cutout')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0
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

path = os.path.join(args.save_root + args.data_name + '_' + args.model+ '_' + str(args.augmentation))
writer = SummaryWriter(log_dir=path)

# Data
print ('The dataset used for training is:                '+ str(args.data_name))
print ('The training mode is:                            '+ str(args.model))
print ('Whether to use data enhancement:                 '+ str(args.augmentation))
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=1)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=args.test_bs, shuffle=False, num_workers=1)

net = Teacher(num_classes=NUM_CLASSES).cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat_a = np.zeros((NUM_CLASSES, NUM_CLASSES))
    conf_mat_b = np.zeros((NUM_CLASSES, NUM_CLASSES))

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
        
        if args.augmentation:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, 0.6)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        else:
            inputs, targets = Variable(inputs), Variable(targets)
        
        _, _, _, _, outputs = net(inputs)
        
        if args.augmentation:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
        
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        
        if args.augmentation:
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

        else:
                conf_mat += losses.confusion_matrix(outputs, targets, NUM_CLASSES)
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
    net.eval()
    PrivateTest_loss = 0
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        test_bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, _, _, _, outputs = net(inputs)
        outputs_avg = outputs.view(test_bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        
        conf_mat += losses.confusion_matrix(outputs_avg, targets, NUM_CLASSES)
        acc = sum([conf_mat[i, i] for i in range(conf_mat.shape[0])])/conf_mat.sum()
        precision = [conf_mat[i, i]/(conf_mat[i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
        mAP = sum(precision)/len(precision)

        recall = [conf_mat[i, i]/(conf_mat[:, i].sum() + 1e-10) for i in range(conf_mat.shape[0])]
        precision = np.array(precision)
        recall = np.array(recall)
        f1 = 2 * precision*recall/(precision+recall + 1e-10)
        F1_score = f1.mean()
        
    return PrivateTest_loss/(batch_idx+1), 100.*acc, 100.* mAP, 100 * F1_score


for epoch in range(0, total_epoch):
	# train one epoch
	train_loss, train_acc, train_mAP, train_F1 = train(epoch)
	# evaluate on testing set
	test_loss, test_acc, test_mAP, test_F1 = test(epoch)

	print("train_loss:  %0.3f, train_acc:  %0.3f, train_mAP:  %0.3f, train_F1:  %0.3f"%(train_loss, train_acc, train_mAP, train_F1))
	print("test_loss:   %0.3f, test_acc:   %0.3f, test_mAP:   %0.3f, test_F1:   %0.3f"%(test_loss, test_acc, test_mAP, test_F1))

	writer.add_scalars('epoch/loss', {'train': train_loss, 'test': test_loss}, epoch)
	writer.add_scalars('epoch/accuracy', {'train': train_acc, 'test': test_acc}, epoch)
	writer.add_scalars('epoch/mAP', {'train': train_mAP, 'test': test_mAP}, epoch)
	writer.add_scalars('epoch/F1', {'train': train_F1, 'test': test_F1}, epoch)

	# save model
	if test_acc > best_acc:
		best_acc = test_acc
		best_mAP = test_mAP
		best_F1 = test_F1
		print ('Saving models......')
		print("best_PrivateTest_acc: %0.3f" % best_acc)
		print("best_PrivateTest_mAP: %0.3f" % best_mAP)
		print("best_PrivateTest_F1: %0.3f" % best_F1)
		state = {
			'tnet': net.state_dict() if use_cuda else net,
			'best_PrivateTest_acc': test_acc,
			'test_mAP': test_mAP,
			'test_F1': test_F1,
			'test_epoch': epoch,
		} 
		if not os.path.isdir(path):
				os.mkdir(path)
		torch.save(state, os.path.join(path,'Best_Teacher_model.t7'))

print("best_PrivateTest_acc: %0.3f" % best_acc)
print("best_PrivateTest_mAP: %0.3f" % best_mAP)
print("best_PrivateTest_F1: %0.3f" % best_F1)
writer.close()
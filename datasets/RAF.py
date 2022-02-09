''' RAF-DB Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import os
import cv2
import random
import torchvision
from torchvision import transforms as transforms
import utils


class RAF_multi_teacher(data.Dataset):
    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split 
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))

        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))

    def __getitem__(self, index):
        
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)
        
        
class RAF(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None, student_norm=None, teacher_norm=None, S_size=44):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.S_size = S_size
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))
        
        
        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
            img = Image.fromarray(img)
            
            if self.S_size == 32:
                img_student = torchvision.transforms.Resize(36)(img)
            elif self.S_size == 24:
                img_student = torchvision.transforms.Resize(28)(img)
            elif self.S_size == 16:
                img_student = torchvision.transforms.Resize(20)(img)
            elif self.S_size == 8:
                img_student = torchvision.transforms.Resize(12)(img)
            else:
                img_student = torchvision.transforms.Resize(48)(img)

            img_student = self.transform(img_student)

            return img_student, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class RAF_student(data.Dataset):
    """`FER2013 Dataset.
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))
        
        
        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = torchvision.transforms.Resize(48)(img)
            img = self.transform(img)
            
            return img, target

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
            img = Image.fromarray(img)
            img = torchvision.transforms.Resize(48)(img)
            img = self.transform(img)

            return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)
        
        
        
class RAF_Online(data.Dataset):
    def __init__(self, split='Training', transform=None, student_norm=None, teacher_norm=None, noise=None):
        self.transform = transform
        self.student_norm = student_norm
        self.teacher_norm = teacher_norm
        self.split = split
        self.noise = noise
        self.data = h5py.File('datasets/RAF_data_100.h5', 'r', driver='core')
        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['train_data_pixel']
            self.train_labels = self.data['train_data_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((12271, 100, 100, 3))
        
        
        else:
            self.PrivateTest_data = self.data['valid_data_pixel']
            self.PrivateTest_labels = self.data['valid_data_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3068, 100, 100, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target

        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

            img_student = self.student_norm(img)
            img_teacher = self.teacher_norm(img)
        
            return img_teacher, img_student, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        
        else:
            return len(self.PrivateTest_data)

# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
from typing import DefaultDict
import numpy as np
import torch
from torch.autograd.grad_mode import enable_grad
import torch.utils
import torch.utils.data
from scipy import io
import yaml
import time
import torch.nn.init as init
from torch.optim import lr_scheduler
import math
import os
import torch.nn.functional as F
import random
import torch.backends.cudnn as cudnn
from  augmentation import RandomFlip,RandomRotate90, Shift,RandomNoise,RandomCrop
from sklearn.metrics import confusion_matrix
def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    # elif hyperparameters['lr_policy'] == 'inv_sqrt':
    # # originally used for Transformer (in Attention is all you need)
    #     def lr_lambda(epoch):
    #         d =  hyperparameters['dim']
    #         warm =  hyperparameters['warm_up_steps']
    #         return d**(-0.5)*min((epoch+1)**(-0.5),(epoch+1)*warm**(-1.5))
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,last_epoch=iterations)
    elif hyperparameters['lr_policy'] == 'zmm':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(epoch):
            decay =  hyperparameters['decay']
            return 1/(1+decay*iterations)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,last_epoch=iterations)    
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'one':
                init.zeros_(m.weight.data)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'xavier_uniform': 
                init.xavier_uniform_(m.weight.data, gain=math.sqrt(2))  #
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def get_config(config):
    with open(config, 'r', encoding = 'utf-8') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_all_datasets(config):
      
    dataset = io.loadmat(config['data_root'])
    source_hsi = dataset['hsi']
    source_map = dataset['map_train']
    target_map = dataset['map_test']

    return source_hsi, source_map,target_map

class HyDataset_all(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, training=True,transform=None, **config):
        """
        Args:
    
        """
        super(HyDataset_all, self).__init__()
        hsi, train_map, test_map = get_all_datasets(config)
        if 'ratio' in config :
            if config['ratio'] is not None:
                train_map = train_ratio(train_map,config['ratio'])
                io.savemat("y_2_seghsi.mat",{"y":train_map})
        hsi = hsi.astype('float32')

        if config['norm']:
            hsi = hsi/np.sqrt(np.sum(hsi**2,axis=2,keepdims=True)+1e-16)

        mean = np.mean(hsi.reshape(-1,hsi.shape[2]),keepdims=True,axis=0).reshape(1,1,hsi.shape[2])
        var = np.sqrt(np.var(hsi.reshape(-1,hsi.shape[2]),keepdims=True,axis=0).reshape(1,1,hsi.shape[2]))
        hsi = (hsi-mean)/var
        self.inputdim = hsi.shape[2]


        class_num = int(np.max(train_map))
        self.class_num = class_num
        train_map = train_map -1 
        train_map[train_map<0] = 255
        test_map = test_map - 1
        test_map[test_map<0] = 255

        
        if config['data_root'] == './data/Houston2018_ln_200.mat':

            self.hsi = [hsi[:,:596,:],hsi[:,596:596*2,:],hsi[:,596*2:596*3,:],hsi[:,596*3:596*4,:]]
            self.train_map = [train_map[:,:596],train_map[:,596:596*2],train_map[:,596*2:596*3],train_map[:,596*3:596*4]]
            self.test_map = [test_map[:,:596],test_map[:,596:596*2],test_map[:,596*2:596*3],test_map[:,596*3:596*4]]
        else:
            self.hsi = [hsi]
            self.train_map = [train_map]
            self.test_map = [test_map]
        # self.hsi_test = [hsi]
        # self.test_map = [test_map]
        self.training = training
        self.transform = [   #RandomCrop(crop_rate=0.1, prob=0.5),
                            
                            RandomFlip(prob=0.5), 
                            # RandomRotate90(prob=0.5),
                            Shift(prob=0.5),
                            
                            # RandomNoise(prob=0.5)
                            
                            # RandomCropOut(crop_rate=0.06, prob=0.5), 
                            # RandomBrightness(bright_range=0.1, prob=0.5), 
                            # RandomNoise(noise_range=10, prob=0.2), 
                           ]
    def __len__(self):
        if self.training:
            return len(self.hsi)
        else:
            return len(self.hsi)
    def get_train_item(self,i):
        hsi = np.asarray(np.copy(self.hsi[i]).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(self.train_map[i]), dtype='int64')
        
        # lidar_2D = lidar_2D.unsqueeze(0)
        
        # prob = random.random()
    
        for func in self.transform:
            hsi, label = func(hsi, label)         
        hsi = torch.from_numpy(hsi.copy())
        label = torch.from_numpy(label.copy())
        # Load the data into PyTorch tensors
         

        return hsi,label
    
    def get_test_item(self,i):
        hsi = np.asarray(np.copy(self.hsi[i]).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(self.test_map[i]), dtype='int64')
        # Load the data into PyTorch tensors
        hsi = torch.from_numpy(hsi)
        # lidar_2D = lidar_2D.unsqueeze(0)
        label = torch.from_numpy(label)        

        return hsi,label
        
    def __getitem__(self,i):
        if self.training == True:
            hsi, label = self.get_train_item(i)
        else:
            hsi, label = self.get_test_item(i)

        return hsi, label

def iouloss(input,target):
    intersection = (input*target).sum(dim=0)
    union = (input + target).sum(dim=0) - intersection
    meaniou = intersection/(union)
    

    return 1 - meaniou.mean()

def diceloss(input,target):
    intersection = (input*target).sum(dim=0)
    union = (input + target).sum(dim=0) 
    dice = 2.*intersection/(union)
    

    return 1 - dice.mean()

def to_one_hot(tensor, n_classes):
        n = tensor.shape[0]
        one_hot = torch.zeros( (n, n_classes)).to("cuda:0").scatter_(1, tensor.view(n, 1), 1)
        return one_hot

def soft_iouloss( input, target,n_classes):
    # logit => N x Classes x H x W
    # target => N x H x W

    N = len(input)

    pred = F.softmax(input,dim=1)


    target = to_one_hot(target,n_classes)
    target = target.float()

    # Numerator Product
    inter = (pred * target).sum(0)
    
    # Denominator
    union = pred + target - (pred * target)
    # Sum over all pixels N x C x H x W => N x C
    union = union.sum(0)
    IOU = inter / (union + 1e-16)
    # Return average loss over classes and batch
    return 1 - IOU.mean()

def confusion(y0,y1,class_num):
    A = torch.zeros(class_num,class_num).cuda()
    for i in range(y0.size(0)):
        A[y0[i].int(),y1[i].int()] += 1  
    return A

def compute_AAK1(y0,y1):
    # OA = torch.sum(x==y)/m
    # AA = 0
    if type(y0) is np.ndarray: 
        cm = confusion_matrix(y0,y1)   
    elif type(y0) is torch.Tensor:
        cm = confusion_matrix(y0.cpu().numpy(),y1.cpu().numpy())
    SA = np.diag(cm)/np.sum(cm,axis=1)
    AA = SA.mean()
    OA = np.sum(np.diag(cm))/np.sum(cm)
    pe = np.sum(np.sum(cm,axis=1,keepdims=True)*np.sum(cm,axis=0,keepdims=True).T)\
        /np.sum(np.sum(cm))**2
    Kappa = (OA-pe)/(1-pe)
    miou = np.mean(np.diag(cm) /( np.sum(cm,0)+ np.sum(cm,1) - np.diag(cm)))
    return OA*100,AA*100,Kappa*100,SA*100, miou*100
def train_ratio(map_train,r):
    m,n = map_train.shape
    map_train = map_train.flatten()
    num_class = int(np.amax(map_train))
    map_train_new = np.zeros_like(map_train)
    for i in range(1,num_class+1):
        pos = np.where(map_train==i)
        lenpos = len(pos[0])
        np.random.seed(2023)
        pos_new = np.random.permutation(pos[0]).flatten()
        map_train_new[pos_new[:int(round(lenpos*r))]] = i

    return map_train_new.reshape(m,n)

class HyDataset(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, hsi, map,transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyDataset, self).__init__()
        self.transform = transform
        self.hsi = hsi
        self.labels = map
        # self.name = hyperparams['dataset']
        self.ignored_labels = hyperparams['ignored_labels']
        self.patch_size = hyperparams['patch_size']
        supervision = hyperparams['supervision']
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(map)
            for l in self.ignored_labels:
                mask[map == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(map)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p-1 and x < hsi.shape[0] - p and y > p-1 and y < hsi.shape[1] - p])
        self.labels = [self.labels[x,y] for x,y in self.indices] 
        # state = np.random.get_state()
        # np.random.shuffle(self.indices)
        # np.random.set_state(state)
        # np.random.shuffle(self.labels)
    # @staticmethod
    # def flip(arrays):
    #     horizontal = np.random.random() > 0.5
    #     vertical = np.random.random() > 0.5
    #     if horizontal:
    #         arrays = np.fliplr(arrays) 
    #     if vertical:
    #         arrays = np.flipud(arrays) 
    #     return arrays

    # @staticmethod
    # def radiation_noise(data, alpha_range=(0.95, 1.05), beta=1/25):
    #     alpha = np.random.uniform(*alpha_range)
    #     noise = np.random.normal(loc=0., scale=1e-3, size=data.shape)
    #     return alpha * data + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        hsi_2D = self.hsi[x1:x2, y1:y2,:]
        label = self.labels[i] - 1 # 类别从0开始
        #flip
        # if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
        #         # Perform data augmentation (only on 2D patches)
        #     hsi_2D = self.flip(hsi_2D)
        # if self.radiation_augmentation and np.random.random() < 0.5:
        #     hsi_2D = self.radiation_noise(hsi_2D)
        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        hsi_2D = np.asarray(np.copy(hsi_2D).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')
        # Load the data into PyTorch tensors
        hsi_2D = torch.from_numpy(hsi_2D)
        label = torch.from_numpy(label)
        # Extract the label if needed      
        return hsi_2D, label
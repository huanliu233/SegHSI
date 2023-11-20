from torch.nn.modules.loss import CrossEntropyLoss

from clusterformer import Clusterformer
from cluterformer_ef_attn_uperhead import Clusterformer_ef_uperhead
from UNet import UNet
from semseg.model_sem.pspnet import PSPNet
from segformer_pytorch.segformer_pytorch.segformer_pytorch import Segformer
from afformer import AFFormer
import torch
import torch.nn as nn
import torch.nn.functional as F 
import os 
from datasetsA import  weights_init,  get_scheduler,iouloss,diceloss,compute_AAK1,soft_iouloss
# import adabound
import math
import numpy as np
from  augmentation_torch import RandomFlip,RandomRotate90, Shift,RandomNoise,RandomCrop
from freenet import FreeNet
from configs.freenet_1_0_muffl import config as muffl
from configs.freenet_1_0_grss2018 import config as houston2018
from configs.freenet_1_0_rsetree import config as rsetree
import ml_collections
from seghsi_for_module_ablation import SegHSI



transform = [Shift(prob=1),
            RandomFlip(prob=0.5), 
            # RandomRotate90(prob=0.5),
            ]
class Clusterformer_trainer(nn.Module):
    def __init__(self,params):
        super(Clusterformer_trainer,self).__init__()
        self.model = Clusterformer(in_chans=params['input_dim'], n_class=params['class_num'],
           n_buckets=params['clusters'], layers=params['depth'], dim = params['dim']) 
        self.model_mom = Clusterformer(in_chans=params['input_dim'], n_class=params['class_num'],
           n_buckets=params['clusters'], layers=params['depth'], dim = params['dim']) 
        # self.model = Clusterformer_ef_uperhead(in_chans=params['input_dim'],n_class=params['class_num'])
        # self.model_mom = Clusterformer_ef_uperhead(in_chans=params['input_dim'],n_class=params['class_num'])
        # self.model = UNet(in_chans=params['input_dim'],n_class=params['class_num'])
        # self.model_mom = UNet(in_chans=params['input_dim'],n_class=params['class_num'])
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.cls_num = params['class_num']

        model_params = list(self.model.parameters())
        self.model_opt = torch.optim.AdamW([p for p in model_params if p.requires_grad],\
                                            lr=params['lr'], \
                                            betas=(0.9, 0.999), \
                                            weight_decay=0.0001, amsgrad =  False)
        # self.model_opt = torch.optim.SGD([p for p in model_params if p.requires_grad],\
        #                                     lr=params['lr'], \
        #                                     momentum=0.9, \
        #                                     weight_decay=0.0001, nesterov=True)
        #get_model_list
        self.model_scheduler = get_scheduler(self.model_opt, params)
        self.lr = params['lr']
        self.params = params
    def forward(self,x):


        return 1


    def update(self,x, y,epoch):
        self.train()
        self.model_opt.zero_grad()
        en_loss = nn.CrossEntropyLoss(label_smoothing=0.0,ignore_index=255)
        # KL_loss = nn.KLDivLoss(reduction="batchmean")
        # mse_loss = nn.MSELoss()
        pred = self.model(x)
        # pred_to_loss = pred.squeeze(0).flatten(1).transpose(0,1)
        self.loss =  en_loss(pred, y) 
        # y_s_f = y.squeeze(0).flatten(0)
        # y_pos = torch.where(y_s_f!=255)
        # # y_pos_select = torch.Tensor(np.random.choice(y_pos[0].cpu().numpy(), y_pos[0].shape[0] // 3,replace=False)).long()
        # pred_to_loss = pred_to_loss[y_pos_select,:]
        # self.loss =  en_loss(pred_to_loss, y_s_f[y_pos_select]) 
        # x1 = x.clone()
        # y1 = y.clone()
        # x2 = x.clone()
        # y2 = y.clone()
        # for func in transform:
        #     x1,y1 = func(x1,y1)
        #     x2,y2 = func(x2,y2)
        # ict_mix = torch.from_numpy(np.random.beta(0.2,0.2,size=(1,1,x1.shape[2],x1.shape[3])).astype('float32')).cuda().detach()
        # pred_mom1 = self.model_mom(x1).squeeze(0).flatten(1).transpose(0,1).softmax(dim=1).detach()
        # pred_mom2 = self.model_mom(x2).squeeze(0).flatten(1).transpose(0,1).softmax(dim=1).detach()
        # pred_mom = pred_mom1*ict_mix.squeeze(0).flatten(1).transpose(0,1)\
        #       + (1-ict_mix.squeeze(0).flatten(1).transpose(0,1))*pred_mom2
  
        # pred = self.model(x1*ict_mix + (1-ict_mix)*x2).squeeze(0).flatten(1).transpose(0,1).softmax(dim=1)
        
        # self.loss += 0.1*var.sum(dim=1).mean()



        pred_mom = self.model_mom(x)
        pred_mom = pred_mom.flatten(2).transpose(1,2).flatten(0,1).detach()
        pred= pred.flatten(2).transpose(1,2).flatten(0,1)

        # self.loss += 0.1*soft_iouloss(pred[y.flatten(0)!=255],y.flatten(0)[y.flatten(0)!=255],self.cls_num)

        pred = pred[y.flatten(0)==255,:]
        pred_mom = pred_mom[y.flatten(0)==255,:]
        var = (pred.softmax(dim=1)-pred_mom.softmax(dim=1))**2
        if self.params['wTrainStra']:
            self.loss += 0.1*var.sum(dim=1).mean()  + 0.01*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1))                                                                                              
        self.loss.backward()
        # if (iter+1)%2 == 0:
        self.model_opt.step()
        # self.model_opt.zero_grad()
        self._update_momentum()
    def _update_momentum(self, m=0.9,epoch=100):
        """Momentum update of the momentum encoder"""
        # if epoch <100:
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m) 
        # elsae:
        #     for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
        #         param_m.data = param_m.data * 0.999 + param_b.data * (1. - 0.999)             
        # for param_b, param_m in zip(self.model_mom.parameters(), self.model.parameters()):
        #     param_m.data = param_m.data * m + parm_b.data * (1. - m) 

    def update_learning_rate1(self, it,epoch, max_iter,max_epoch,warm_up_epoch):

        if it + max_iter*(epoch)<warm_up_epoch*max_iter:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = (self.lr-1e-6)/warm_up_epoch/max_iter * (it + max_iter*(epoch) ) + 1e-6
        else:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = self.lr * (math.cos((it + max_iter*(epoch-warm_up_epoch) ) * (math.pi / max_iter/(max_epoch-warm_up_epoch))) + 1)/2  + 1e-6  
        print(param_group['lr'])     

 
    def print_loss(self, x_2D,y):

        self.eval() 
        class_pred=  self.model(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA,AA,Kappa,SA,miou = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        class_pred=  self.model_mom(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        # correct = sum(torch.eq(y.view(-1), class_pred.view(-1))*(y.view(-1)!=255))
        OA_m,AA_m,Kappa_m,SA_m,miou_m = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        return OA,AA,Kappa,SA,miou,OA_m,AA_m,Kappa_m,SA_m,miou_m


def get_optimizer_grouped_parameters(
    model,
    learning_rate, weight_decay, 
    layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "clf" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
   # initialize lrs for every layer
    # num_layers = model.config.num_hidden_layers
    # layers =[model.patch_embed]+ [model.pos_embed_temporal] + [model.pos_embed_spatial] + \
    #       [model.blocks]  + [model.clf]
# [getattr(model, 'pos_embed_spatial'),getattr(model, 'pos_embed_temporal')] + [getattr(model, 'pos_embed_temporal')] +
    layers =  \
          [getattr(model, 'patch_embed')]+ list(getattr(model, 'blocks')) +  [getattr(model, 'norm')] 
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


class freenet_trainer(nn.Module):
    def __init__(self,params):
        super(freenet_trainer,self).__init__()
        self.model = FreeNet(config=muffl) 
        self.model_mom = FreeNet(config=muffl) 
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.cls_num = params['class_num']

        model_params = list(self.model.parameters())
        self.model_opt = torch.optim.AdamW([p for p in model_params if p.requires_grad],\
                                            lr=params['lr'], \
                                            betas=(0.9, 0.999), \
                                            weight_decay=0.0001, amsgrad =  False)
        self.model_scheduler = get_scheduler(self.model_opt, params)
        self.lr = params['lr']
    def forward(self,x):


        return 1


    def update(self,x, y,epoch):
        self.train()
        self.model_opt.zero_grad()
        en_loss = nn.CrossEntropyLoss(label_smoothing=0.0,ignore_index=255)
        pred = self.model(x)
        self.loss =  en_loss(pred, y) 

        pred_mom = self.model_mom(x)
        pred_mom = pred_mom.flatten(2).transpose(1,2).flatten(0,1).detach()
        pred= pred.flatten(2).transpose(1,2).flatten(0,1)
        pred = pred[y.flatten(0)==255,:]
        pred_mom = pred_mom[y.flatten(0)==255,:]
        var = (pred.softmax(dim=1)-pred_mom.softmax(dim=1))**2
        # if epoch<=100:
        #     self.loss += 0.01/100*epoch*var.sum(dim=1).mean()  + 0.001/100*epoch*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1)) 
        # else:
        self.loss += 0.01*var.sum(dim=1).mean()  + 0.001*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1))                                                                                                  
        self.loss.backward()
        self.model_opt.step()
        self._update_momentum()
    def _update_momentum(self, m=0.9,epoch=100):
        """Momentum update of the momentum encoder"""
        # if epoch <100:
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m) 
        # elsae:
        #     for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
        #         param_m.data = param_m.data * 0.999 + param_b.data * (1. - 0.999)             
        # for param_b, param_m in zip(self.model_mom.parameters(), self.model.parameters()):
        #     param_m.data = param_m.data * m + parm_b.data * (1. - m) 

    def update_learning_rate1(self, it,epoch, max_iter,max_epoch,warm_up_epoch):

        if it + max_iter*(epoch)<warm_up_epoch*max_iter:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = (self.lr-1e-6)/warm_up_epoch/max_iter * (it + max_iter*(epoch) ) + 1e-6
        else:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = self.lr * (math.cos((it + max_iter*(epoch-warm_up_epoch) ) * (math.pi / max_iter/(max_epoch-warm_up_epoch))) + 1)/2  + 1e-6  
        print(param_group['lr'])     

 
    def print_loss(self, x_2D,y):

        self.eval() 
        class_pred=  self.model(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA,AA,Kappa,SA,miou = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        class_pred=  self.model_mom(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        correct = sum(torch.eq(y.view(-1), class_pred.view(-1))*(y.view(-1)!=255))
        OA_m,AA_m,Kappa_m,SA_m,miou_m = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        return OA,AA,Kappa,SA,miou,OA_m,AA_m,Kappa_m,SA_m,miou_m
    

class UNet_trainer(nn.Module):
    def __init__(self,params):
        super(UNet_trainer,self).__init__()
        self.model = UNet(in_chans=params['input_dim'],n_class=params['class_num'])
        self.model_mom = UNet(in_chans=params['input_dim'],n_class=params['class_num'])
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.cls_num = params['class_num']

        model_params = list(self.model.parameters())
        self.model_opt = torch.optim.AdamW([p for p in model_params if p.requires_grad],\
                                            lr=params['lr'], \
                                            betas=(0.9, 0.999), \
                                            weight_decay=0.0001, amsgrad =  False)
        self.model_scheduler = get_scheduler(self.model_opt, params)
        self.lr = params['lr']
        self.params = params
    def forward(self,x):

        return 1


    def update(self,x, y,iter):
        self.train()
        self.model_opt.zero_grad()
        en_loss = nn.CrossEntropyLoss(label_smoothing=0.0,ignore_index=255)
        pred = self.model(x)
        self.loss =  en_loss(pred, y) 
        pred_mom = self.model_mom(x)
        pred_mom = pred_mom.flatten(2).transpose(1,2).flatten(0,1).detach()
        pred= pred.flatten(2).transpose(1,2).flatten(0,1)

        # self.loss += 0.1*soft_iouloss(pred[y.flatten(0)!=255],y.flatten(0)[y.flatten(0)!=255],self.cls_num)

        pred = pred[y.flatten(0)==255,:]
        pred_mom = pred_mom[y.flatten(0)==255,:]
        var = (pred.softmax(dim=1)-pred_mom.softmax(dim=1))**2
        if self.params['wTrainStra']:
            self.loss += 0.1*var.sum(dim=1).mean()  + 0.01*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1))
        self.loss = self.loss                                                                                                  
        self.loss.backward()
        # if (iter+1)%2 == 0:
        self.model_opt.step()
        # self.model_opt.zero_grad()
        self._update_momentum()
    def _update_momentum(self, m=0.9,epoch=100):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m) 

    def update_learning_rate1(self, it,epoch, max_iter,max_epoch,warm_up_epoch):

        if it + max_iter*(epoch)<warm_up_epoch*max_iter:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = (self.lr-1e-6)/warm_up_epoch/max_iter * (it + max_iter*(epoch) ) + 1e-6
        else:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = self.lr * (math.cos((it + max_iter*(epoch-warm_up_epoch) ) * (math.pi / max_iter/(max_epoch-warm_up_epoch))) + 1)/2  + 1e-6  
        print(param_group['lr'])     

 
    def print_loss(self, x_2D,y):

        self.eval() 
        class_pred=  self.model(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA,AA,Kappa,SA,miou = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        class_pred=  self.model_mom(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA_m,AA_m,Kappa_m,SA_m,miou_m = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        return OA,AA,Kappa,SA,miou,OA_m,AA_m,Kappa_m,SA_m,miou_m

class PSPNet_trainer(nn.Module):
    def __init__(self,params):
        super(PSPNet_trainer,self).__init__()
        self.model = PSPNet(inputdim=params['input_dim'],classes=params['class_num'])
        self.model_mom = PSPNet(inputdim=params['input_dim'],classes=params['class_num'])
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.cls_num = params['class_num']

        model_params = list(self.model.parameters())
        self.model_opt = torch.optim.AdamW([p for p in model_params if p.requires_grad],\
                                            lr=params['lr'], \
                                            betas=(0.9, 0.999), \
                                            weight_decay=0.0001, amsgrad =  False)
        self.model_scheduler = get_scheduler(self.model_opt, params)
        self.lr = params['lr']
        self.params = params
    def forward(self,x):

        return 1


    def update(self,x, y,iter):
        self.train()
        self.model_opt.zero_grad()
        en_loss = nn.CrossEntropyLoss(label_smoothing=0.0,ignore_index=255)
        pred = self.model(x)
        self.loss =  en_loss(pred, y) 
        pred_mom = self.model_mom(x)
        pred_mom = pred_mom.flatten(2).transpose(1,2).flatten(0,1).detach()
        pred= pred.flatten(2).transpose(1,2).flatten(0,1)

        # self.loss += 0.1*soft_iouloss(pred[y.flatten(0)!=255],y.flatten(0)[y.flatten(0)!=255],self.cls_num)

        pred = pred[y.flatten(0)==255,:]
        pred_mom = pred_mom[y.flatten(0)==255,:]
        var = (pred.softmax(dim=1)-pred_mom.softmax(dim=1))**2
        if self.params['wTrainStra']:
            self.loss += 0.01*var.sum(dim=1).mean()  #+ 0.01*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1))                                                                                                 
        self.loss.backward()
        # if (iter+1)%2 == 0:
        self.model_opt.step()
        # self.model_opt.zero_grad()
        self._update_momentum()
    def _update_momentum(self, m=0.9,epoch=100):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m) 

    def update_learning_rate1(self, it,epoch, max_iter,max_epoch,warm_up_epoch):

        if it + max_iter*(epoch)<warm_up_epoch*max_iter:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = (self.lr-1e-6)/warm_up_epoch/max_iter * (it + max_iter*(epoch) ) + 1e-6
        else:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = self.lr * (math.cos((it + max_iter*(epoch-warm_up_epoch) ) * (math.pi / max_iter/(max_epoch-warm_up_epoch))) + 1)/2  + 1e-6  
        print(param_group['lr'])     

 
    def print_loss(self, x_2D,y):

        self.eval() 
        class_pred=  self.model(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA,AA,Kappa,SA,miou = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        class_pred=  self.model_mom(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA_m,AA_m,Kappa_m,SA_m,miou_m = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        return OA,AA,Kappa,SA,miou,OA_m,AA_m,Kappa_m,SA_m,miou_m

class segformer_trainer(nn.Module):
    def __init__(self,params):
        super(segformer_trainer,self).__init__()
        self.model = Segformer(channels=params['input_dim'],num_classes=params['class_num']) 
        self.model_mom = Segformer(channels=params['input_dim'],num_classes=params['class_num'])
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.cls_num = params['class_num']

        model_params = list(self.model.parameters())
        self.model_opt = torch.optim.AdamW([p for p in model_params if p.requires_grad],\
                                            lr=params['lr'], \
                                            betas=(0.9, 0.999), \
                                            weight_decay=0.0001, amsgrad =  False)
        self.model_scheduler = get_scheduler(self.model_opt, params)
        self.lr = params['lr']
        self.params = params
    def forward(self,x):

        return 1


    def update(self,x, y,iter):
        self.train()
        self.model_opt.zero_grad()
        en_loss = nn.CrossEntropyLoss(label_smoothing=0.0,ignore_index=255)
        pred = self.model(x)
        self.loss =  en_loss(pred, y) 
        pred_mom = self.model_mom(x)
        pred_mom = pred_mom.flatten(2).transpose(1,2).flatten(0,1).detach()
        pred= pred.flatten(2).transpose(1,2).flatten(0,1)

        # self.loss += 0.1*soft_iouloss(pred[y.flatten(0)!=255],y.flatten(0)[y.flatten(0)!=255],self.cls_num)

        pred = pred[y.flatten(0)==255,:]
        pred_mom = pred_mom[y.flatten(0)==255,:]
        var = (pred.softmax(dim=1)-pred_mom.softmax(dim=1))**2
        if self.params['wTrainStra']:
            self.loss += 0.1*var.sum(dim=1).mean()  # + 0.001*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1))                                                                                                 
        self.loss.backward()
        # if (iter+1)%2 == 0:
        self.model_opt.step()
        # self.model_opt.zero_grad()
        self._update_momentum()
    def _update_momentum(self, m=0.9,epoch=100):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m) 

    def update_learning_rate1(self, it,epoch, max_iter,max_epoch,warm_up_epoch):

        if it + max_iter*(epoch)<warm_up_epoch*max_iter:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = (self.lr-1e-6)/warm_up_epoch/max_iter * (it + max_iter*(epoch) ) + 1e-6
        else:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = self.lr * (math.cos((it + max_iter*(epoch-warm_up_epoch) ) * (math.pi / max_iter/(max_epoch-warm_up_epoch))) + 1)/2  + 1e-6  
        print(param_group['lr'])     

 
    def print_loss(self, x_2D,y):

        self.eval() 
        class_pred=  self.model(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA,AA,Kappa,SA,miou = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        class_pred=  self.model_mom(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA_m,AA_m,Kappa_m,SA_m,miou_m = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        return OA,AA,Kappa,SA,miou,OA_m,AA_m,Kappa_m,SA_m,miou_m

class afformer_trainer(nn.Module):
    def __init__(self,params):
        super(afformer_trainer,self).__init__()
        self.model = AFFormer(in_chans=params['input_dim'],num_classes=params['class_num']) 
        self.model_mom = AFFormer(in_chans=params['input_dim'],num_classes=params['class_num'])         
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.cls_num = params['class_num']

        model_params = list(self.model.parameters())
        self.model_opt = torch.optim.AdamW([p for p in model_params if p.requires_grad],\
                                            lr=params['lr'], \
                                            betas=(0.9, 0.999), \
                                            weight_decay=0.01, amsgrad =  False)
        self.model_scheduler = get_scheduler(self.model_opt, params)
        self.lr = params['lr']
        self.params = params
    def forward(self,x):

        return 1


    def update(self,x, y,epoch):
        self.train()
        self.model_opt.zero_grad()
        en_loss = nn.CrossEntropyLoss(label_smoothing=0.0,ignore_index=255)
        pred = self.model(x)
        self.loss =  en_loss(pred, y) 
        pred_mom = self.model_mom(x)
        pred_mom = pred_mom.flatten(2).transpose(1,2).flatten(0,1).detach()
        pred= pred.flatten(2).transpose(1,2).flatten(0,1)

        # self.loss += 0.1*soft_iouloss(pred[y.flatten(0)!=255],y.flatten(0)[y.flatten(0)!=255],self.cls_num)

        pred = pred[y.flatten(0)==255,:]
        pred_mom = pred_mom[y.flatten(0)==255,:]
        var = (pred.softmax(dim=1)-pred_mom.softmax(dim=1))**2
        if self.params['wTrainStra']:
            # if epoch<200:
            self.loss += 0.01*var.sum(dim=1).mean() #+ 0.01*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1))#+= epoch/300*0.1*var.sum(dim=1).mean()   #+ 0.01*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1)))
            # else:
            #     self.loss += 1*var.sum(dim=1).mean()   #+ 0.01*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1))                                                                                               
        self.loss.backward()
        # if (iter+1)%2 == 0:
        self.model_opt.step()
        # self.model_opt.zero_grad()
        self._update_momentum()
    def _update_momentum(self, m=0.9,epoch=100):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m) 

    def update_learning_rate1(self, it,epoch, max_iter,max_epoch,warm_up_epoch):

        if it + max_iter*(epoch)<warm_up_epoch*max_iter:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = (self.lr-1e-6)/warm_up_epoch/max_iter * (it + max_iter*(epoch) ) + 1e-6
        else:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = self.lr * (math.cos((it + max_iter*(epoch-warm_up_epoch) ) * (math.pi / max_iter/(max_epoch-warm_up_epoch))) + 1)/2  + 1e-6  
        print(param_group['lr'])     

 
    def print_loss(self, x_2D,y):

        self.eval() 
        class_pred=  self.model(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA,AA,Kappa,SA,miou = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        class_pred=  self.model_mom(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA_m,AA_m,Kappa_m,SA_m,miou_m = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        return OA,AA,Kappa,SA,miou,OA_m,AA_m,Kappa_m,SA_m,miou_m

class SegHSI_trainer(nn.Module):
    def __init__(self,params):
        super(SegHSI_trainer,self).__init__()
        self.model = SegHSI(in_chans=params['input_dim'], n_class=params['class_num'],
           n_buckets=params['clusters'], layers=params['depth'], dim = params['dim'],attn=params['attn'],mlp=params['mlp']) 
        self.model_mom = SegHSI(in_chans=params['input_dim'], n_class=params['class_num'],
           n_buckets=params['clusters'], layers=params['depth'], dim = params['dim'],attn=params['attn'],mlp=params['mlp']) 
        # self.model = Clusterformer_ef_uperhead(in_chans=params['input_dim'],n_class=params['class_num'])
        # self.model_mom = Clusterformer_ef_uperhead(in_chans=params['input_dim'],n_class=params['class_num'])
        # self.model = UNet(in_chans=params['input_dim'],n_class=params['class_num'])
        # self.model_mom = UNet(in_chans=params['input_dim'],n_class=params['class_num'])
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.cls_num = params['class_num']

        model_params = list(self.model.parameters())
        self.model_opt = torch.optim.AdamW([p for p in model_params if p.requires_grad],\
                                            lr=params['lr'], \
                                            betas=(0.9, 0.999), \
                                            weight_decay=0.0001, amsgrad =  False)
        # self.model_opt = torch.optim.SGD([p for p in model_params if p.requires_grad],\
        #                                     lr=params['lr'], \
        #                                     momentum=0.9, \
        #                                     weight_decay=0.0001, nesterov=True)
        #get_model_list
        self.model_scheduler = get_scheduler(self.model_opt, params)
        self.lr = params['lr']
        self.params = params
    def forward(self,x):


        return 1


    def update(self,x, y,epoch):
        self.train()
        self.model_opt.zero_grad()
        en_loss = nn.CrossEntropyLoss(label_smoothing=0.0,ignore_index=255)


        pred = self.model(x)
        self.loss =  en_loss(pred, y) 
 
        pred_mom = self.model_mom(x)
        pred_mom = pred_mom.flatten(2).transpose(1,2).flatten(0,1).detach()
        pred= pred.flatten(2).transpose(1,2).flatten(0,1)

        pred = pred[y.flatten(0)==255,:]
        pred_mom = pred_mom[y.flatten(0)==255,:]
        var = (pred.softmax(dim=1)-pred_mom.softmax(dim=1))**2
        if self.params['wTrainStra']:
            self.loss += 0.1*var.sum(dim=1).mean()  + 0.01*iouloss(pred.softmax(dim=1),pred_mom.softmax(dim=1))                                                                                              
        self.loss.backward()
        # if (iter+1)%2 == 0:
        self.model_opt.step()
        # self.model_opt.zero_grad()
        self._update_momentum()
    def _update_momentum(self, m=0.9,epoch=100):
        """Momentum update of the momentum encoder"""
        # if epoch <100:
        for param_b, param_m in zip(self.model.parameters(), self.model_mom.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m) 

    def update_learning_rate1(self, it,epoch, max_iter,max_epoch,warm_up_epoch):

        if it + max_iter*(epoch)<warm_up_epoch*max_iter:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = (self.lr-1e-6)/warm_up_epoch/max_iter * (it + max_iter*(epoch) ) + 1e-6
        else:
            for param_group in self.model_opt.param_groups:
                param_group['lr'] = self.lr * (math.cos((it + max_iter*(epoch-warm_up_epoch) ) * (math.pi / max_iter/(max_epoch-warm_up_epoch))) + 1)/2  + 1e-6  
        print(param_group['lr'])     

 
    def print_loss(self, x_2D,y):

        self.eval() 
        class_pred=  self.model(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA,AA,Kappa,SA,miou = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        class_pred=  self.model_mom(x_2D)
        class_pred = torch.argmax(F.softmax(class_pred,dim=1), dim=1)
        OA_m,AA_m,Kappa_m,SA_m,miou_m = compute_AAK1( y.view(-1)[y.view(-1)!=255]
            ,class_pred.view(-1)[y.view(-1)!=255])

        return OA,AA,Kappa,SA,miou,OA_m,AA_m,Kappa_m,SA_m,miou_m
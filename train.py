from datasetsA import get_all_datasets, get_config, HyDataset_all
import argparse  # 传输参数值的命令
from trainer import Clusterformer_trainer
import torch.backends.cudnn as cudnn
import torch
import torch.utils.data as data
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy import io
import time
import torch.nn.functional as F
import os
torch.cuda.set_device('cuda:0')
torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./data/**.mat', help="data_root")
parser.add_argument('--clusters', type=int, default=64, help="cluters")
parser.add_argument('--batch_size', type=int, default=1, help="batch_size")
parser.add_argument('--batch_size_test', type=int, default=1, help="batch_size_test")
parser.add_argument('--dim', type=int, default=64, help="dim")
parser.add_argument('--depth', type=int, default=4, help="depth")
parser.add_argument('--lr', type=float, default=0.01, help="lr")
parser.add_argument('--beta1', type=float, default=0.9, help="beta1")
parser.add_argument('--beta2', type=float, default=0.999, help="beta2")
parser.add_argument('--norm', type=bool, default=False, help="norm")
config = parser.parse_args()
config = vars(config)
if config['data_root'] == './data/Houston2018_ln_200.mat':
    config['batch_size'] = 2
    config['batch_size_test'] = 4
train_dataset = HyDataset_all(training=True,transform=None, **config)
test_dataset = HyDataset_all(training=False,transform=None, **config)
config['class_num'] = train_dataset.class_num
config['input_dim'] = train_dataset.inputdim

model_name = "clusterformer"
dataname = os.path.split(config['data_root'])[-1].split('.')[0]
model_dir_path = './train_cluster/%s/'%dataname 
os.makedirs(model_dir_path, exist_ok=True)
checkpoint_model_file = os.path.join(model_dir_path, 'tmp.pth')
checkpoint_optim_file = os.path.join(model_dir_path, 'tmp.optim')
final_model_file      = os.path.join(model_dir_path, 'final.pth')
log_file              = os.path.join(model_dir_path, 'log.txt')


train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'],pin_memory=True, shuffle=True,drop_last = False)
test_loader = data.DataLoader(test_dataset, batch_size=config['batch_size_test'], pin_memory=False, shuffle=False,drop_last = False)
check_epoch = 0
start = time.time()


trainer = Clusterformer_trainer(config)
trainer.cuda()
check_epoch = 0
max_epoch = 500
warm_up_epoch = 50
acc = 0
for epoch in range(check_epoch,max_epoch):
    max_iter = len(train_loader)
    loss = 0
    for it, (hsi_2Ds,y) in enumerate(train_loader):
        
        trainer.update_learning_rate1(it,epoch,max_iter,max_epoch,warm_up_epoch)


        hsi_2Ds, y= hsi_2Ds.cuda().detach(), y.cuda().detach()    
        trainer.update(hsi_2Ds,y,epoch=epoch)
        loss = trainer.loss
        if it % 30==0:
            print("epoch %04d---iter %04d" % (epoch,it))
            print("loss = %04f" %loss)

    total_correct_num = 0 
    total_num = 0
    loss = 0 
    for its, (hsi_2D,y) in enumerate(test_loader):
        with torch.no_grad():
            hsi_2D, y = hsi_2D.cuda().detach(), y.cuda().detach()
            OA,AA,Kappa,SA,miou,OA_m,AA_m,Kappa_m,SA_m,miou_m =  trainer.print_loss(hsi_2D,y)

    mtx0 = '|epoch:%d' %epoch +'\n'
    mtx1 = '| OA:%.2f AA:%.2f Kappa:%.2f mIoU: %0.2f\n'% (OA,
        AA,Kappa,miou) 
    mtx2 = '|SA:'+str(SA)+'\n'
    mtx3 = '| OA_m:%.2f AA_m:%.2f Kappa_m:%.2f, mIoU_m: %0.2f\n'% (OA_m,
        AA_m,Kappa_m,miou_m) 
    mtx4 = '|SA:'+str(SA_m)+'\n' + '\n'
    with open(log_file, 'a') as appender:
        appender.write(mtx0)
        appender.write(mtx1)
        appender.write(mtx2)
        appender.write(mtx3)
        appender.write(mtx4)
    if OA>90 and acc < OA and OA> OA_m:
        acc = float(OA)
        print('| saving check point model file... ', end='')
        checkpoint_epoch_name = model_dir_path + model_name  +'_epo' + str(epoch) + '_OA_' + str(format(acc,'.2f'))  +'.pth'
        torch.save(trainer.model.state_dict(), checkpoint_epoch_name)
    elif OA_m>90 and acc < OA_m and OA< OA_m:
        acc = float(OA_m)
        print('| saving check point model file... ', end='')
        checkpoint_epoch_name = model_dir_path + model_name  +'_epo' + str(epoch) + '_OA_m_' + str(format(acc,'.2f'))  +'.pth'
        torch.save(trainer.model_mom.state_dict(), checkpoint_epoch_name)

    print("Testing Accuracy: %02f and %02f" % (OA,OA_m))



import os
import argparse

from tqdm import tqdm
import pandas as pd

import glob

from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import torch.backends.cudnn as cudnn
# optim三
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import torchvision.transforms as transforms
from PIL import Image
# dataloader
from torch.utils.data import DataLoader
from dataset import Get_SDataset,Get_UDataset
# model
from pslpt import Double_Pyramid_Former as Pyramid_Former
import random
from random import randrange
# loss
from losses import ssim_loss_vi
from loss_vif import fusion_loss_vif
from utils import cal_psnr, compute_ssim
import cv2
import numpy as np

from itertools import cycle

seed = 555
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark  = False


device = torch.device('cuda:0')
l1_loss = torch.nn.L1Loss().to(device)
uns_loss = fusion_loss_vif().to(device)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='pyramid_model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--gamma', default=0.5, type=int)
    parser.add_argument('--sbatch_size', default=2, type=int)
    parser.add_argument('--s1batch_size', default=2, type=int)
    parser.add_argument('--ubatch_size', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float)
    parser.add_argument('--weight', default=[1,1,0.0001, 0.0002], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()

    return args

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args,train_loader_ir,train_loader_ir1,train_loader_u,model,criterion_ssim_vi,optimizer):
    losses = AverageMeter()
    losses_ssim_ir = AverageMeter()
    losses_ssim_vi = AverageMeter()
    model.train()
    for i, (batch_s, batch_s1, batch_u) in tqdm(enumerate(zip(cycle(train_loader_ir),cycle(train_loader_ir1), train_loader_u))):
        nf,ff,clear = Variable(batch_s[0]), Variable(batch_s[1]), Variable(batch_s[2])
        u1,u2 = Variable(batch_u[0]), Variable(batch_u[1])
        nf1,ff1,clear1 = Variable(batch_s1[0]), Variable(batch_s1[1]), Variable(batch_s1[2])

        nf = nf.to(device)
        ff = ff.to(device)
        clear = clear.to(device)
        nf1 = nf1.to(device)
        ff1 = ff1.to(device)
        clear1 = clear1.to(device)
        u1 = u1.to(device)
        u2 = u2.to(device)
        
        optimizer.zero_grad()
        
        # supervised forward
        out_clear,out_nf,out_ff,s_log_sig = model(nf,ff,second_stage=True)
        out_clear1,out_nf1,out_ff1,s_log_sig1 = model(nf1,ff1,second_stage=True)

        mfif_loss = 2*criterion_ssim_vi(out_clear, clear) + l1_loss(out_nf, nf)  + l1_loss(out_ff, ff)
        meif_loss = 2*criterion_ssim_vi(out_clear1, clear1) + l1_loss(out_nf1, nf1)  + l1_loss(out_ff1, ff1)
        super_loss = mfif_loss + 0.01*meif_loss # naive training
        
        wau_out_clear, wau_out_u1, wau_out_u2, u_log_sig = model(u1, u2,second_stage=True)

        wau_recon_loss = l1_loss(u1, wau_out_u1) + l1_loss(u2,wau_out_u2)
        unsuper_loss = 0.04*uns_loss(u1,u2,wau_out_clear) + wau_recon_loss
        loss_ssim_ir = super_loss + unsuper_loss
        total_loss = super_loss + 0.1*unsuper_loss
        print("Loss: {:.2e} || MFIF Loss: {:.2e} || MEIF Loss: {:.2e}".format(total_loss.item(), mfif_loss.item(), meif_loss.item()))
        print("Unsuper Loss: {:.2e} || U_recon_Loss: {:.2e}".format(unsuper_loss.item(), wau_recon_loss.item()))
        losses_ssim_ir.update(loss_ssim_ir.item(), nf.size(0))

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-4, norm_type=2)
        optimizer.step()
        model.zero_grad()

def test(scheduler_f, model, test_loader_ir):
    avg_psnr = 0.0
    avg_ssim = 0.0
    torch.set_grad_enabled(False)
    epoch = scheduler_f.last_epoch
    model.eval()
    print('\nEvaluation:')
     
    for i, (ir,vi,clear)  in tqdm(enumerate(test_loader_ir), total=len(test_loader_ir)):
        with torch.no_grad():
            ir = ir.to(device)
            vi = vi.to(device)
            clear = clear.to(device)
            factor = 32
            if ir.shape[-2]%factor != 0:
                new_h = ir.shape[-2] - ir.shape[-2]%factor
                ir = ir[:,:,:new_h,:]
                vi = vi[:,:,:new_h,:]
                clear = clear[:,:,:new_h,:]
            if ir.shape[-1]%factor != 0:
                new_w = ir.shape[-1] - ir.shape[-1]%factor
                ir = ir[:,:,:,:new_w]
                vi = vi[:,:,:,:new_w]
                clear = clear[:,:,:,:new_w]
            
            with torch.no_grad():
                out,_ = model.test(vi,ir,second_stage=True)
                out = out.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,0)
                clear = clear.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,0)
                
                avg_psnr += cal_psnr(out,clear)
                avg_ssim += compute_ssim(out,clear)
                
    avg_psnr = avg_psnr / len(test_loader_ir)
    avg_ssim = avg_ssim / len(test_loader_ir)
    
    if avg_psnr >= ckt['psnr']:
        ckt['psnr_epoch'] = epoch
        ckt['psnr'] = avg_psnr
    if avg_ssim >= ckt['ssim']:
        ckt['ssim_epoch'] = epoch
        ckt['ssim'] = avg_ssim
    print("===> Avg.PSNR: {:.4f} dB || ssim: {:.4f} || Best.PSNR: {:.4f} dB || Best_PSNR_Epoch: {}|| Best.SSIM: {:.4f} || Best_SSIM_Epoch: {}"
          .format(avg_psnr, avg_ssim, ckt['psnr'], ckt['psnr_epoch'], ckt['ssim'], ckt['ssim_epoch']))
    torch.set_grad_enabled(True)

def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    # supervised train_data
    train_dir_f = "" # near focus
    train_dir_b = "" # far focus
    train_dir_g = "" # gt
    train_name_list = os.listdir(train_dir_f)

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
    dataset_train_ir = Get_SDataset(train_dir_f,train_dir_b,train_dir_g,train_name_list,
                                                  transform=transform_train)

    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.sbatch_size)
    # supervised train_data 2
    train_dir_f = "" # under exposure
    train_dir_b = "" # over exposure
    train_dir_g = "" # gt
    train_name_list = os.listdir(train_dir_f)

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
    dataset_train_ir1 = Get_SDataset(train_dir_f,train_dir_b,train_dir_g,train_name_list,
                                                  transform=transform_train)

    train_loader_ir1 = DataLoader(dataset_train_ir1,
                              shuffle=True,
                              batch_size=args.s1batch_size)
    # ivf train_data
    train_dir_u2 = "" # ir
    train_dir_u1 = "" # vi
    train_name_list_u = os.listdir(train_dir_u1)

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
    dataset_train_u = Get_UDataset(train_dir_u1,train_dir_u2,train_name_list_u,
                                                  transform=transform_train)

    train_loader_u = DataLoader(dataset_train_u,
                              shuffle=True,
                              batch_size=args.ubatch_size)
    # test data
    test_dir_f = "" 
    test_dir_b = "" 
    test_dir_g = "" 
    test_name_list = os.listdir(test_dir_f)
    dataset_test_ir = Get_SDataset(test_dir_f,test_dir_b,test_dir_g,test_name_list,is_patch=False,
                                                  transform=transform_train)
    test_loader_ir = DataLoader(dataset_test_ir,
                              shuffle=True,
                              batch_size=1)
    

    model = Pyramid_Former(num_channel=1, base_filter=48).to(device)
    model_path = "./models/pyramid_model/two_fuse_rule.pth"
    model.load_state_dict(torch.load(model_path))

    criterion_ssim_vi = ssim_loss_vi
    milestones = []
    for i in range(1, args.epochs+1):
        if i == 200:
            milestones.append(i)
        if i == 300:
            milestones.append(i)

    optimizer = optim.Adam([{'params':model.fl_former2.parameters()},{'params':model.fh_former2.parameters()},
                            {'params':model.fh2_former2.parameters()},{'params':model.fh4_former2.parameters()}], lr=args.lr,
                           betas=args.betas, eps=args.eps)
    scheduler_f = lrs.MultiStepLR(optimizer, milestones, args.gamma)

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))
        model.zero_grad()
        train(args,train_loader_ir,train_loader_ir1,train_loader_u,model,criterion_ssim_vi,optimizer)
        scheduler_f.step()
        if (epoch+1) % 1 == 0:
            test(scheduler_f, model, test_loader_ir)
            torch.save(model.state_dict(), 'models/%s/model_{}.pth'.format(epoch+1) %args.name)

if __name__ == '__main__':
    ckt = {'psnr_epoch':0, 'ssim_epoch':0, 'ssim':0.0, 'psnr':0.0} 
    i_iter = 0
    main()
    


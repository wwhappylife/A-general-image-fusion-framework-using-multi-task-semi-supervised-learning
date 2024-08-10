import numpy as np
import os
import torch

import glob
import time

import torchvision.transforms as transforms
from thop import clever_format
from torch.utils.data import DataLoader, Dataset
from math import exp
import torch.nn.functional as F

from pslpt import Double_Pyramid_Former as Pyramid_Former
from tqdm import tqdm
import argparse
import cv2
from utils import print_network

device = torch.device('cuda:0')


class GetDataset(Dataset):
    def __init__(self, training_dir_ir, training_dir_vi, ir_name_list, vi_name_list, transform=None):
        super(GetDataset, self).__init__()
        ir_name_list.sort()
        vi_name_list.sort()
        self.training_dir_ir = training_dir_ir
        self.training_dir_vi = training_dir_vi
        self.ir_name_list = ir_name_list
        self.vi_name_list = vi_name_list
        self.transform = transform

    def __getitem__(self, index):
        ir = cv2.imread(self.training_dir_ir + self.ir_name_list[index])
        ir = cv2.cvtColor(ir, cv2.COLOR_BGR2YCrCb)

        vi = cv2.imread(self.training_dir_vi + self.vi_name_list[index])
        vi = cv2.cvtColor(vi, cv2.COLOR_BGR2YCrCb)

        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = self.transform
            ir = tran(ir)
            vi = tran(vi)
            return ir,vi, self.ir_name_list[index]

    def __len__(self):
        return len(self.ir_name_list)
        
# for mri-pet task, set is_second_stage="True"
# training_dir_ir = "your mri directory"
# training_dir_vi = "your pet directory" 

# for mri-ct task, set is_second_stage="True"
# training_dir_ir = "your ct directory"
# training_dir_vi = "your mri directory" 

# for mri-spect task, set is_second_stage="True"
# training_dir_ir = "your mri directory"
# training_dir_vi = "your pet directory" 

# for ivf task, set is_second_stage="True"
# training_dir_ir = "your ir directory"
# training_dir_vi = "your vi directory" 

# for mef task, set is_second_stage="false"
# training_dir_ir = "your underexposure image directory"
# training_dir_vi = "your overexposure image directory" 

# for mff task, set is_second_stage="false"
# training_dir_ir = "your far-focus image directory"
# training_dir_vi = "your near-focus image directory"

training_dir_ir = "/home/wangwu/mfif_dataset/MFI-WHU30/B/" # Lytro20;
ir_name_list = os.listdir(training_dir_ir) 
print(ir_name_list)
training_dir_vi = "/home/wangwu/mfif_dataset/MFI-WHU30/A/" # Lytro20
vi_name_list = os.listdir(training_dir_vi) 

transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])

dataset_test_dir = GetDataset(training_dir_ir, training_dir_vi, ir_name_list, vi_name_list,
                                                  transform=transform_train)
test_loader = DataLoader(dataset_test_dir,
                              shuffle=False,
                              batch_size=1)

model = Pyramid_Former(num_channel=1, base_filter=48).to(device)
print_network(model)
model_path = "./models/pyramid_model/two_fuse_rule.pth" 
model.load_state_dict(torch.load(model_path))

def fusion(is_second_stage):
    
    fl = 0.0
    pa = 0.0
    tic = time.time()
    for i, (ir,vi, name)  in tqdm(enumerate(test_loader), total=len(test_loader)):
        

        ir = ir.to(device)
        vi = vi.to(device)
        factor = 16
        if ir.shape[-2]%factor != 0:
            new_h = ir.shape[-2] - ir.shape[-2]%factor
            ir = ir[:,:,:new_h,:]
            vi = vi[:,:,:new_h,:]
        if ir.shape[-1]%factor != 0:
            new_w = ir.shape[-1] - ir.shape[-1]%factor
            ir = ir[:,:,:,:new_w]
            vi = vi[:,:,:,:new_w]
            
        iry = ir[:,0:1, :, :]
        ircr = ir[:,1:2, :, :]
        ircb = ir[:,2:3, :, :]
        viy = vi[:,0:1, :,:]
        vicr = vi[:,1:2, :,:]
        vicb = vi[:,2:3, :,:]

        cr = torch.cat((vicr,ircr),dim=0)
        cb = torch.cat((ircb,vicb),dim=0)
       
        EPS = 1e-6
        w_cr = (torch.abs(cr) + EPS) / torch.sum(torch.abs(cr) + EPS, dim=0)
        w_cb = (torch.abs(cb) + EPS) / torch.sum(torch.abs(cb) + EPS, dim=0)
        fcr = torch.sum(w_cr * cr, dim=0, keepdim=True).clamp(-1, 1)
        fcb = torch.sum(w_cb * cb, dim=0, keepdim=True).clamp(-1, 1)

        if i <= 220:
            
            with torch.no_grad():
                model.eval()
                # to test on mfif and meif task, set second_stage=False; to test on IVF and MMIF task, set second_stage=True
                out,mask = model.test(iry, viy,second_stage=False)
                out = torch.clamp(out,0,1)
            if is_second_stage:
                out = torch.cat((out,vicr,vicb),dim=1)
            else:
                out = torch.cat((out,fcr, fcb),dim=1)
            result = np.squeeze(out.detach().permute(0,2,3,1).cpu().numpy())
            result = cv2.cvtColor(result, cv2.COLOR_YCrCb2BGR)
            result = result * 255
            result = np.clip(result,0,255)
            result = result.astype(np.uint8)

            print(name[0])
            cv2.imwrite('./result_mfi/'+name[0], result)
           
        
    toc = time.time()
    print('end {}{}'.format(i // 10, i % 10), ', time:{}'.format(toc - tic))
    fl, pa = clever_format([fl,pa],"%.3f")
    print(fl, pa)



if __name__ == '__main__':
    is_second_stage = False
    fusion(is_second_stage)

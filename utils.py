import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import mm
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
#from skimage.measure import compare_ssim
import random
import numpy
#import xlrd
import os

def cal_thres(I,I1,I2,t=0.1):
    thres_value = torch.ones_like(I).cuda()
    thres_value = thres_value * t
    I_max = choose_max(I1,I2)
    confid_map = torch.abs(I-I_max)/(I_max+1e-6)
    thres_map = torch.le(confid_map, thres_value)
    return thres_map

def choose_max(im1, im2):
    im1 = im1.unsqueeze(-1)
    im2 = im2.unsqueeze(-1)
    im = torch.cat((im1,im2),dim=-1)
    max_im,_ = torch.max(im,dim=-1)
    return max_im

def crop_image_pair(factor,ir,vi):
    if ir.shape[-2]%factor != 0:
        new_h = ir.shape[-2] - ir.shape[-2]%factor    
        ir = ir[:,:,:new_h,:]
        vi = vi[:,:,:new_h,:]
    if ir.shape[-1]%factor != 0:
        new_w = ir.shape[-1] - ir.shape[-1]%factor
        ir = ir[:,:,:,:new_w]
        vi = vi[:,:,:,:new_w]
    return ir,vi

def mef_norm(I1,I2):
    mean_I1 = torch.mean(torch.mean(I1,dim=2,keepdim=True),dim=3,keepdim=True)
    mean_I2 = torch.mean(torch.mean(I2,dim=2,keepdim=True),dim=3,keepdim=True)
    std_I1 = torch.std(I1,dim=[2,3],keepdim=True)
    std_I2 = torch.std(I2,dim=[2,3],keepdim=True)
    mean = (mean_I1 + mean_I2)/2
    std = (std_I1 + std_I2)/2
    I1 = (I1 - mean_I1)/(std_I1+1e-6) * std + mean
    I2 = (I2 - mean_I2)/(std_I2+1e-6) * std + mean
    return I1,I2

def ins_norm(I):
    I = (I - torch.mean(torch.mean(I,dim=2,keepdim=True),dim=3,keepdim=True))/(torch.std(I,dim=[2,3],keepdim=True)+1e-6)
    return I

def out_norm(I,I1,I2,type):
    mean_I1 = torch.mean(torch.mean(I1,dim=2,keepdim=True),dim=3,keepdim=True)
    mean_I2 = torch.mean(torch.mean(I2,dim=2,keepdim=True),dim=3,keepdim=True)
    std_I1 = torch.std(I1,dim=[2,3],keepdim=True)+1e-6
    std_I2 = torch.std(I2,dim=[2,3],keepdim=True)+1e-6
    if type=='1':
        I = I*std_I1 + mean_I1
    elif type=='2':
        I = I*std_I2 + mean_I2
    else:
        I = I*(std_I1+std_I2)/2 + (mean_I1+mean_I2)/2
    return I

def CE(rgb_img, increment):
    img = rgb_img * 1.0
    img_min = img.min(axis=2)
    img_max = img.max(axis=2)
    img_out = img
    
    #获取HSL空间的饱和度和亮度
    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value/2.0
    
    # s = L<0.5 ? s1 : s2
    mask_1 = L < 0.5
    
    s1 = delta/(value + 1e-6)
    s2 = delta/(2 - value)
    s = s1 * mask_1 + s2 * (1 - mask_1)
    
    # 增量大于0，饱和度指数增强
    if increment >= 0 :
        # alpha = increment+s > 1 ? alpha_1 : alpha_2
        temp = increment + s
        mask_2 = temp >  1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        
        alpha = 1/alpha -1 
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
        
    # 增量小于0，饱和度线性衰减
    else:
        alpha = increment
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
    
    img_out = img_out/255.0
    
    # RGB颜色上下限处理(小于0取0，大于1取1)
    mask_3 = img_out  < 0 
    mask_4 = img_out  > 1
    img_out = img_out * (1-mask_3)
    img_out = img_out * (1-mask_4) + mask_4
    
    return img_out

def noise_augment2(img1, img2):
    img1 = img1 + torch.rand_like(img1, dtype=torch.float)*0.01
    img2 = img2 + torch.rand_like(img2, dtype=torch.float)*0.01
    return img1,img2

def mixup_data(ms, pan, gt, alpha=1.0, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam=1.
    batch_size = ms.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_ms = lam * ms + (1 - lam) * ms[index,:]
    mixed_pan = lam * pan + (1 - lam) * pan[index,:]
    mixed_gt = lam * gt + (1 - lam) * gt[index,:]
    return mixed_ms, mixed_pan, mixed_gt

def mixup_data_wo_gt(ms, pan, alpha=1.0, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam=1.
    batch_size = ms.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_ms = lam * ms + (1 - lam) * ms[index,:]
    mixed_pan = lam * pan + (1 - lam) * pan[index,:]
    return mixed_ms, mixed_pan

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def compute_ergas(out, gt):
    num_spectral = out.shape[-1]
    out = np.reshape(out, (-1, num_spectral)) 
    gt = np.reshape(gt, (-1, num_spectral))
    diff = gt - out
    mse = np.mean(np.square(diff), axis=0)
    gt_mean = np.mean(gt, axis=0)
    mse = np.reshape(mse, (num_spectral,1))
    gt_mean = np.reshape(gt_mean, (num_spectral,1))
    ergas = 100/4*np.sqrt(np.mean(mse/(gt_mean**2+1e-6)))
    return ergas

def compute_sam(im1, im2):
    num_spectral = im1.shape[-1]
    im1 = np.reshape(im1, (-1, num_spectral))
    im2 = np.reshape(im2, (-1, num_spectral))
    mole = np.sum(np.multiply(im1, im2), axis=1)
    im1_norm = np.sqrt(np.sum(np.square(im1), axis=1))
    im2_norm = np.sqrt(np.sum(np.square(im2), axis=1))
    deno = np.multiply(im1_norm, im2_norm)
    
    sam = np.rad2deg(np.arccos((mole)/(deno+1e-7)))
    sam = np.mean(sam)
    return sam

def compute_ssim(im1, im2):
    single_ssim = compare_ssim(im1, im2)
    return single_ssim

def compute_ssim_rgb(im1, im2):
    multi_ssim = (compare_ssim(im1[:,:,0], im2[:,:,0]) + compare_ssim(im1[:,:,1], im2[:,:,1]) + compare_ssim(im1[:,:,2], im2[:,:,2]))/3
    return multi_ssim

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    

def cal_psnr(im1, im2):
    im1 = np.reshape(im1, (-1,1))
    im2 = np.reshape(im2, (-1,1))
    diff = im1 - im2

    mse = np.mean(np.square(diff), axis=0)
    return np.mean(10 * np.log10(1/mse))
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()
'''
def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)
'''
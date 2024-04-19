
from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
#from utils.utils_color import RGB_HSV, RGB_YCbCr
#from models.loss_ssim import ssim
from MEFSSIM.lossfunction import MEFSSIM
import torchvision.transforms.functional as TF
from math import exp
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def Contrast(img1, img2, window_size=11, channel=1):
    window = create_window(window_size, channel)    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq

    
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_GT_Grad1(nn.Module):
    def __init__(self):
        super(L_GT_Grad1, self).__init__()
        self.sobelconv=Sobelxy()
        #self.mefssim = MEFSSIM(channel=1)
    def forward(self, image_A, image_B, source1, source2):
        Loss_gradient = 0.0
        for i in range(image_A.shape[1]):
            gradient_A = self.sobelconv(image_A[:,i,:,:].unsqueeze(1))
            gradient_A = gradient_A
            #gradient_A = torch.clamp(gradient_A,0,1)
            gradient_B = self.sobelconv(image_B[:,i,:,:].unsqueeze(1))
            gradient_source1 = self.sobelconv(source1[:,i,:,:].unsqueeze(1))
            gradient_source2 = self.sobelconv(source2[:,i,:,:].unsqueeze(1))
            gradient = torch.max(gradient_B,gradient_source1)
            gradient = torch.max(gradient,gradient_source2)
            Loss_gradient += F.l1_loss(gradient_A, gradient_B)
            #Loss_gradient += self.mefssim(gradient_A, gradient_B)
        return Loss_gradient

class L_GT_Grad(nn.Module):
    def __init__(self):
        super(L_GT_Grad, self).__init__()
        self.sobelconv=Sobelxy()
    def forward(self, image_A, image_B):
        Loss_gradient = 0.0
        for i in range(image_A.shape[1]):
            gradient_A = self.sobelconv(image_A[:,i,:,:].unsqueeze(1))
            gradient_A = gradient_A
            gradient_A = torch.clamp(gradient_A,0,1)
            gradient_B = self.sobelconv(image_B[:,i,:,:].unsqueeze(1))
            Loss_gradient += F.l1_loss(gradient_A, gradient_B)
        return Loss_gradient
'''
class L_GT_Grad(nn.Module):
    def __init__(self):
        super(L_GT_Grad, self).__init__()
        self.sobelconv=Sobelxy()
    def forward(self, image_A, image_B):
        Loss_gradient = 0.0
        for i in range(image_A.shape[1]):
            gradient_A = self.sobelconv(image_A[:,i,:,:].unsqueeze(1))
            gradient_A = gradient_A
            #gradient_A = torch.clamp(gradient_A,0,1)
            gradient_B = self.sobelconv(image_B[:,i,:,:].unsqueeze(1))
            mask = torch.where(gradient_A>gradient_B,gradient_A,gradient_B)
            Loss_gradient += F.l1_loss(gradient_A*mask, gradient_B*mask)
            #Loss_gradient += F.l1_loss(gradient_A, gradient_B)
        return Loss_gradient
'''
class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient
        
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):        
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss

class fusion_loss_med(nn.Module):
    def __init__(self):
        super(fusion_loss_med, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        # image_A represents MRI image
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 100 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 50 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss
    
class fusion_loss_cddfuse(nn.Module):
    def __init__(self):
        super(fusion_loss_cddfuse, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        # image_A represents MRI image
        loss_l1 = 2*self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 5 * self.L_Grad(image_A, image_B, image_fused)
        fusion_loss = loss_l1 + loss_gradient
        return fusion_loss


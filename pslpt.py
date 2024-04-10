#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:34:22 2020

@author: wangwu
"""


from email.mime import base
import torch
import torch.nn as nn
from einops import rearrange
import scipy.io as sio
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
from timm.models.layers import trunc_normal_, DropPath
from math import exp
from torch.autograd import Variable
from network_swin2sr import BasicLayer
import os


class Mask_Fuse(nn.Module):
    def __init__(self, num_spectral,window_size):
        super(Mask_Fuse, self).__init__()
        
        self.num_spectral = num_spectral
        self.pad = nn.ReflectionPad2d(1)
        self.proj_in = nn.Conv2d(num_spectral, num_spectral, 3, 1, 0, bias=True)
        self.proj_out = nn.Conv2d(num_spectral, 2, 3, 1, 0, bias=True)
    
        self.glo = BasicLayer(dim=num_spectral,
                                     input_resolution=(128, 128),
                                     depth=2,
                                     num_heads=6,
                                     window_size=window_size,
                                     mlp_ratio=3,
                                     qkv_bias=True,
                                     drop=0, attn_drop=0,
                                     drop_path=0,
                                     norm_layer=nn.LayerNorm,
                                     downsample=None,
                                     use_checkpoint=False)
        self.in_ln = nn.LayerNorm(num_spectral)
        self.out_ln = nn.LayerNorm(num_spectral)
        
    def forward(self, I1,I2, gs=False):
        I = self.proj_in(self.pad(I1-I2))
        B,C,H,W = I.shape
        I = rearrange(I, 'B C H W -> B (H W) C')
        I = self.out_ln(self.glo(self.in_ln(I),(H,W)))
        I = rearrange(I, 'B (H W) C-> B C H W',H=H,W=W)
        I = self.proj_out(self.pad(I))
        if not gs:
            I = F.softmax(I,dim=1)
        else:
            I = F.gumbel_softmax(I, tau=1, hard=False, eps=1e-8, dim=-1)

        LM = F.softmax(I/0.2,dim=1)
        M1 = I[:,1,:,:].unsqueeze(1)
        M2 = I[:,0,:,:].unsqueeze(1)
        I = I1*M1+I2*M2
        M = torch.cat((M1,M2),dim=1)
        return I,LM[:,1,:,:].unsqueeze(1)

class Former(nn.Module):
    def __init__(self, base_filter, window_size=2, depth=2):
        super(Former, self).__init__()

        self.pad = nn.ReflectionPad2d(1)
        self.in_conv = nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False)
        self.out_conv = nn.Conv2d(base_filter, base_filter, 3, 1, 0, bias=False)
        self.g = BasicLayer(dim=base_filter,
                                     input_resolution=(128, 128),
                                     depth=depth,
                                     num_heads=6,
                                     window_size=window_size,
                                     mlp_ratio=3,
                                     qkv_bias=True,
                                     drop=0, attn_drop=0,
                                     drop_path=0,
                                     norm_layer=nn.LayerNorm,
                                     downsample=None,
                                     use_checkpoint=False)
        self.in_ln = nn.LayerNorm(base_filter)
        self.out_ln = nn.LayerNorm(base_filter)
# 
    def forward(self, I, norm=False):
        B,C,H,W = I.shape
        I0 = I
        I = self.in_conv(self.pad(I))
        I = rearrange(I, 'B C H W -> B (H W) C')
        if norm:
            I = self.out_ln(self.g(self.in_ln(I),(H,W)))
        else:
            I = self.g(I,(H,W))
        I = rearrange(I, 'B (H W) C-> B C H W',H=H,W=W)
        I = self.out_conv(self.pad(I))
        return I+I0

class Encoder(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Encoder, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.pool = torch.nn.MaxPool2d(3,stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.eformer1 = Former(base_filter, window_size=2)
        self.eformer2 = Former(base_filter, window_size=2)
        self.eformer4 = Former(base_filter, window_size=2)
        self.eformer8 = Former(base_filter, window_size=2)
        self.in_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_channel, base_filter, 3, 1, 0, bias=False))
    def forward(self, I):
        I = self.in_conv(I)
        I = self.eformer1(I,True)
        I2 = self.pool(self.pad(I))
        I2 = self.eformer2(I2)
        I4 = self.pool(self.pad(I2))
        I4 = self.eformer4(I4)
        I8 = self.pool(self.pad(I4))
        I8 = self.eformer8(I8)
        Ires = I - self.up(I2)
        I2res = I2 - self.up(I4) 
        I4res = I4 - self.up(I8) 
        return Ires,I2res,I4res,I8

class Decoder(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Decoder, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dformer1 = Former(base_filter, window_size=2)
        self.dformer2 = Former(base_filter, window_size=2)
        self.dformer4 = Former(base_filter, window_size=2)
        self.dformer8 = Former(base_filter, window_size=2)
        self.in_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_channel, base_filter, 3, 1, 0, bias=False))
        self.out_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(base_filter, num_channel, 3, 1, 0, bias=False))
    def forward(self, Ires,I2res,I4res,I8):
        I4 = I4res + self.up(self.dformer8(I8))
        I2 = I2res + self.up(self.dformer4(I4))
        I = Ires + self.up(self.dformer2(I2)) 
        emb = I
        I = self.dformer1(I)
        I = self.out_conv(I)
        return I,emb

class Double_Encoder(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Double_Encoder, self).__init__()
        self.encoder1 = Encoder(num_channel,base_filter)
        self.encoder2 = Encoder(num_channel,base_filter)
    def forward(self, I1,I2):
        I1res,I1res2,I1res4,I18 = self.encoder1(I1)
        I2res,I2res2,I2res4,I28 = self.encoder2(I2)
        return I1res,I1res2,I1res4,I18,I2res,I2res2,I2res4,I28

class Learned_Fuse(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Learned_Fuse, self).__init__()
        self.fl_former = Mask_Fuse(base_filter, window_size=2)
        self.fh_former = Mask_Fuse(base_filter, window_size=2)
        self.fh2_former = Mask_Fuse(base_filter, window_size=2)
        self.fh4_former = Mask_Fuse(base_filter, window_size=2)
    def forward(self, I1res,I1res2,I1res4,I18,I2res,I2res2,I2res4,I28):
        I8,ml = self.fl_former(I18,I28)
        Ires, mh = self.fh_former(I1res,I2res)
        Ires2, mh2 = self.fh2_former(I1res2,I2res2)
        Ires4, mh1 = self.fh4_former(I1res4,I2res4)
        return Ires,Ires2,Ires4,I8

class Double_Pyramid_Former(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Double_Pyramid_Former, self).__init__()
        self.encoder1 = Encoder(num_channel,base_filter)
        self.encoder2 = Encoder(num_channel,base_filter)
        self.decoder = Decoder(num_channel,base_filter)
        self.fl_former = Mask_Fuse(base_filter, window_size=2)
        self.fh_former = Mask_Fuse(base_filter, window_size=2)
        self.fh2_former = Mask_Fuse(base_filter, window_size=2)
        self.fh4_former = Mask_Fuse(base_filter, window_size=2)

        self.fl_former2 = Mask_Fuse(base_filter, window_size=2)
        self.fh_former2 = Mask_Fuse(base_filter, window_size=2)
        self.fh2_former2 = Mask_Fuse(base_filter, window_size=2)
        self.fh4_former2 = Mask_Fuse(base_filter, window_size=2)
        
    def forward(self, I1,I2,second_stage=False, gs=False):
        
        I1res,I1res2,I1res4,I18 = self.encoder1(I1)
        I2res,I2res2,I2res4,I28 = self.encoder2(I2)
        # decoder
        rI1,_ = self.decoder(I1res,I1res2,I1res4,I18)
        rI2,_ = self.decoder(I2res,I2res2,I2res4,I28)
        if not second_stage:
            I8,ml = self.fl_former(I18,I28,gs)
            Ires, mh = self.fh_former(I1res,I2res,gs)
            Ires2, mh2 = self.fh2_former(I1res2,I2res2,gs)
            Ires4, mh1 = self.fh4_former(I1res4,I2res4,gs)
        else:
            I8,ml = self.fl_former2(I18,I28,gs)
            Ires, mh = self.fh_former2(I1res,I2res,gs)
            Ires2, mh2 = self.fh2_former2(I1res2,I2res2,gs)
            Ires4, mh1 = self.fh4_former2(I1res4,I2res4,gs)
        F,emb = self.decoder(Ires,Ires2,Ires4,I8)
        return F,rI1,rI2,emb
    
    def test(self,I1,I2,second_stage=False, gs=False):
        I1res,I1res2,I1res4,I18 = self.encoder1(I1)
        I2res,I2res2,I2res4,I28 = self.encoder2(I2)
        
        if not second_stage:
            I8,ml = self.fl_former(I18,I28,gs)
            Ires, mh = self.fh_former(I1res,I2res,gs)
            Ires2, mh2 = self.fh2_former(I1res2,I2res2,gs)
            Ires4, mh1 = self.fh4_former(I1res4,I2res4,gs)
        else:
            I8,ml = self.fl_former2(I18,I28,gs)
            Ires, mh = self.fh_former2(I1res,I2res,gs)
            Ires2, mh2 = self.fh2_former2(I1res2,I2res2,gs)
            Ires4, mh1 = self.fh4_former2(I1res4,I2res4,gs)

        F,emb = self.decoder(Ires,Ires2,Ires4,I8)
        return F,I28[:,25,:,:].unsqueeze(1)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import pdb


class UNet3D(nn.Module):
    def __init__(self, opt):
        super(UNet3D, self).__init__()
        
        ### encoder
        self.ec0 = self.encoder(3, 64, kernel_size=(7,7,7), stride=(2,2,2), padding=(3,3,3), bias=False, batchnorm=False) 
        self.ec1 = self.encoder(64, 128, kernel_size=(5,5,5), stride=(2,2,2), padding=(2,2,2), bias=False, batchnorm=True) 
        self.ec2 = self.encoder(128, 256, kernel_size=(5,5,5), stride=(2,2,2), padding=(2,2,2), bias=False, batchnorm=True)
        self.ec3 = self.encoder(256, 512, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False, batchnorm=True)
        self.ec4 = self.encoder(512, 512, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False, batchnorm=True)

        ### decoder
        self.dc4 = self.decoder((1, 8, 8), 512, 512, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        self.dc3 = self.decoder((1, 16, 16), 512 + 512, 512, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True) 
        self.dc2 = self.decoder((2, 32, 32), 512 + 256, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        self.dc1 = self.decoder((4, 64, 64), 256 + 128, 128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True) 
        self.dc0 = self.decoder((8, 128, 128), 128 + 64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True) 
        
        ### predictors
        self.clip_p = nn.Conv3d(64 + 3, 3, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.mask_p = nn.Sequential(
                        nn.Conv3d(64 + 3, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False),
                        nn.Sigmoid()
                        )
        
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, size, in_channels, out_channels, kernel_size, stride=1, padding=0,
                bias=True, batchnorm=False, mode='trilinear'):
        if batchnorm:
            layer = nn.Sequential(
                nn.Upsample(size=size, mode=mode),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2)
                )
        else:
            layer = nn.Sequential(
                nn.Upsample(size=size, mode=mode),
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                )
        return layer

    def forward(self, x):
        ### encoder
        e0 = self.ec0(x)  # 64 x 64
        e1 = self.ec1(e0) # 32 x 128
        e2 = self.ec2(e1) # 16 x 256
        e3 = self.ec3(e2) # 8  x 512
        e4 = self.ec4(e3) # 4  x 512

        ### decoder
        d4 = torch.cat((self.dc4(e4), e3), 1)
        del e4, e3
        d3 = torch.cat((self.dc3(d4), e2), 1)
        del d4, e2
        d2 = torch.cat((self.dc2(d3), e1), 1)
        del d3, e1
        d1 = torch.cat((self.dc1(d2), e0), 1)
        del d2, e0
        d0 = torch.cat((self.dc0(d1), x), 1)
        del d1, x

        ### predictor
        clip_pred = self.clip_p(d0)
        mask_pred = self.mask_p(d0)
        del d0

        return clip_pred, mask_pred
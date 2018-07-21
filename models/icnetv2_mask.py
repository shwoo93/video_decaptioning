import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import pdb


class ICNetResidual3D(nn.Module):
    def __init__(self, opt):
        super(ICNetResidual3D, self).__init__()
        
        ### encoder
        self.ec0 = self.encoder(3, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False, batchnorm=True)
        self.ec1 = self.encoder(64, 128, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False, batchnorm=True)

        ### bottleneck
        self.bt0 = self.encoder(128, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        self.bt1 = self.encoder(256, 256, kernel_size=(3,3,3), stride=(1,1,1), dilation=(1,2,2), padding=(1,2,2), bias=False, batchnorm=True)
        self.bt2 = self.encoder(256, 256, kernel_size=(3,3,3), stride=(1,1,1), dilation=(1,4,4), padding=(1,4,4), bias=False, batchnorm=True)
        self.bt3 = self.encoder(256, 256, kernel_size=(3,3,3), stride=(1,1,1), dilation=(1,8,8), padding=(1,8,8), bias=False, batchnorm=True)

        ### decoder
        self.dc1 = self.decoder((4, 64, 64), 256, 128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        self.dc0 = self.decoder((8, 128, 128), 128 + 64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        
        ### predictors
        self.clip_p = nn.Conv3d(64 + 3, 3, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=0,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias),
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

        ### bottleneck
        bt0 = self.bt0(e1) # 32 x 256
        bt1 = self.bt1(bt0)# 32 x 256
        bt2 = self.bt2(bt1)# 32 x 256
        bt3 = self.bt3(bt2)# 32 x 256

        ### decoder
        d1 = torch.cat((self.dc1(bt3),e0),1)
        del bt3, e0
        d0 = torch.cat((self.dc0(d1),x),1)
        del d1, x

        clip = self.clip_p(d0)
        del d0

        if self.scaledown:
            clip = torch.clamp(clip, min=0, max=1.0)

        return clip, mask

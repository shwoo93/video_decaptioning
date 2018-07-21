import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import pdb

class GatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, padding=0, bias=False, type='3d'):
        super(GatedConvolution, self).__init__()
        assert type in ['2d', '3d']

        if type == '3d':
            self.phi = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU())

            self.gate = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias),
                        nn.Sigmoid())
        elif type == '2d':
            self.phi = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())

            self.gate = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias),
                        nn.ReLU())

    def forward(self, x):
        phi = self.phi(x)
        gate = self.gate(x)
        return phi * gate


class GatedUpConvolution(nn.Module):
    def __init__(self, size, in_channels, out_channels, kernel_size, stride, padding, bias, mode='trilinear', type='3d'):
        super(GatedUpConvolution, self).__init__()
        assert type in ['2d', '3d']

        if type == '3d':
            self.phi = nn.Sequential(
                        nn.Upsample(size=size, mode=mode),
                        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                        nn.BatchNorm3d(out_channels),
                        nn.LeakyReLU(0.2)
                        )
            self.gate = nn.Sequential(
                        nn.Upsample(size=size, mode=mode),
                        nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                        nn.Sigmoid()
                        )
        elif type == '2d':
            self.phi = nn.Sequential(
                        nn.Upsample(size=size, mode=mode),
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2)
                        )
            self.gate = nn.Sequential(
                        nn.Upsample(size=size, mode=mode),
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                        nn.Sigmoid()
                        )

    def forward(self, x):
        phi = self.phi(x)
        gate = self.gate(x)
        return phi * gate

class ICNetDeepGate2D(nn.Module):
    def __init__(self, opt):
        super(ICNetDeepGate2D, self).__init__()
        self.opt = opt

        ### encoder
        self.ec0 = GatedConvolution(3, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False, type='2d')
        self.ec1 = GatedConvolution(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False, type='2d')  
        self.ec2 = GatedConvolution(128, 256, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False, type='2d')

        ### temporal convs
        self.tc0 = GatedConvolution(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False, type='2d')
        self.tc1 = GatedConvolution(128,128, kernel_size=(3,3), stride=(1,1), padding=(1,1),bias=False, type='2d')

        ### bottleneck
        self.bt0 = GatedConvolution(256, 256, kernel_size=(3,3), stride=(1,1), dilation=(1,1), padding=(1,1), bias=False, type='2d')
        self.bt1 = GatedConvolution(256, 256, kernel_size=(3,3), stride=(1,1), dilation=(2,2), padding=(2,2), bias=False, type='2d')
        self.bt2 = GatedConvolution(256, 256, kernel_size=(3,3), stride=(1,1), dilation=(4,4), padding=(4,4), bias=False, type='2d')

        ### decoder
        self.dc2 = GatedUpConvolution((32, 32), 256 + 256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False, 
                                    mode='bilinear', type='2d')
        self.dc1 = GatedUpConvolution((64, 64), 256 + 128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False,
                                    mode='bilinear', type='2d')
        self.dc0 = GatedUpConvolution((128, 128), 128 + 64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False,
                                    mode='bilinear', type='2d')

        self.clip_diff_p = nn.Conv2d(64, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):

        ### encoder
        e0 = self.ec0(x)
        e1 = self.ec1(e0)
        e2 = self.ec2(e1)

        ### temporal convs
        t0 = self.tc0(e0)
        t1 = self.tc1(e1)

        ### bottleneck
        bt0 = self.bt0(e2)  
        bt1 = self.bt1(bt0) 
        bt2 = self.bt2(bt1) 

        # pdb.set_trace()
        ### decoder
        d2 = self.dc2(torch.cat((bt2, e2),1)) # 16 --> 32
        d1 = self.dc1(torch.cat((d2, t1), 1)) # 32 --> 64
        d0 = self.dc0(torch.cat((d1, t0), 1))

        clip_pred = x[:,:,4,:,:].unsqueeze(2) + self.clip_diff_p(d0) # 64 --> 3

        return torch.sigmoid(clip_pred)
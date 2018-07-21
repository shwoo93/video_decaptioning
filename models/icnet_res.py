import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from .non_local_block import NonLocalBlock
import pdb


class ICNetResidual3D(nn.Module):
    def __init__(self, opt):
        super(ICNetResidual3D, self).__init__()
        
        ### encoder
        self.ec0 = self.encoder(3, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False, batchnorm=True)
        self.ec1 = self.encoder(64, 128, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False, batchnorm=True)
        self.ec2 = self.encoder(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False, batchnorm=True)

        ### bottleneck)
        self.bt0 = self.encoder(256, 256, kernel_size=(3,3,3), stride=(1,1,1), dilation=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        self.bt1 = self.encoder(256, 256, kernel_size=(3,3,3), stride=(1,1,1), dilation=(1,2,2), padding=(1,2,2), bias=False, batchnorm=True)
        self.bt2 = self.encoder(256, 256, kernel_size=(3,3,3), stride=(1,1,1), dilation=(1,4,4), padding=(1,4,4), bias=False, batchnorm=True)

        ### decoder
        self.dc2 = self.decoder((8, 32, 32), 256 + 256, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        self.dc1 = self.decoder((8, 64, 64), 256 + 128, 128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        self.dc0 = self.decoder((8, 128, 128), 128 + 64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True)
        
        ### predictors
        self.clip_diff_p = nn.Conv3d(64, 3, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        e2 = self.ec2(e1) # 16 x 256

        ### bottleneck
        bt0 = self.bt0(e2) # 16 x 256
        bt1 = self.bt1(bt0)# 16 x 256
        bt2 = self.bt2(bt1)# 16 x 256

        ### decoder
        d2 = self.dc2(torch.cat((bt2, e2),1))
        d1 = self.dc1(torch.cat((d2,  e1),1))
        d0 = self.dc0(torch.cat((d1,  e0),1))

        clip_pred = x + self.clip_diff_p(d0)

        # return clip_pred
        # return torch.tanh(clip_pred)*0.5
        return torch.clamp(clip_pred, 0, 1)
        #return torch.sigmoid(clip_pred)




class ICNetResidual2D(nn.Module):
    def __init__(self, opt):
        super(ICNetResidual2D, self).__init__()

        ### temporal convolutions
        self.tc0 = nn.Conv3d(3, 32, kernel_size=(3,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)
        self.bn0 = nn.BatchNorm3d(32)
        self.relu0 = nn.ReLU()
        self.tc1 = nn.Conv3d(32,32, kernel_size=(3,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()     
        ### encoder
        self.ec0 = self.encoder(32, 64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False, batchnorm=True) #64 
        self.ec1 = self.encoder(64, 128, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False, batchnorm=True) #32
        self.ec2 = self.encoder(128, 256, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False, batchnorm=True) #16

        ### bottleneck
        self.bt0 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True)
        self.bt1 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,2,2), padding=(0,2,2), bias=False, batchnorm=True)
        self.bt2 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,4,4), padding=(0,4,4), bias=False, batchnorm=True)

        ### decoder
        self.dc2 = self.decoder((1, 32, 32), 256 + 256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 32
        self.dc1 = self.decoder((1, 64, 64), 256 + 128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 64
        self.dc0 = self.decoder((1, 128, 128), 128 + 64, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 128
        
        ### predictors
        self.clip_diff_p = nn.Conv3d(64, 3, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        ### temporal convs
        t0 = self.relu0(self.bn0(self.tc0(x))) # 7 --> 3
        t1 = self.relu1(self.bn1(self.tc1(t0))) # 3 --> 1

        ### encoder
        e0 = self.ec0(t1)  # 64x(128) --> 64x(64)
        e1 = self.ec1(e0) # 64x(64) --> 128x(32)
        e2 = self.ec2(e1) # 128x(32) --> 256x(16)

        ### bottleneck
        bt0 = self.bt0(e2)  # 256x(16)
        bt1 = self.bt1(bt0) # 256x(16)
        bt2 = self.bt2(bt1) # 256x(16)

        ### decoder
        d2 = self.dc2(torch.cat((bt2, e2),1)) # 16 --> 32
        d1 = self.dc1(torch.cat((d2, e1),1)) # 32 --> 64
        d0 = self.dc0(torch.cat((d1, e0),1))  # 64 --> 128

        clip_pred = x[:,:,3,:,:].unsqueeze(2) + self.clip_diff_p(d0)  # 64 --> 3

        # return clip_pred
        # return torch.tanh(clip_pred)*0.5
        return torch.clamp(clip_pred, 0, 1)
        # return torch.sigmoid(clip_pred)


class ICNetResidual2Dt(nn.Module):
    def __init__(self, opt):
        super(ICNetResidual2Dt, self).__init__()
        
        ### encoder
        self.nl = opt.nl
        self.diff = opt.diff
        self.ec0 = self.encoder(3, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False, batchnorm=True) #64 
        if self.nl:
            self.enl0 = NonLocalBlock(64)
        self.ec1 = self.encoder(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False, batchnorm=True) #32
        if self.nl:
            self.enl1 = NonLocalBlock(128)
        self.ec2 = self.encoder(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1), bias=False, batchnorm=True) #16

        ### temporal convs
        self.tc0 = self.encoder(64, 64, kernel_size=(5,3,3), stride=(1,1,1), padding=(0,1,1),bias=False, batchnorm=True)
        self.tc1 = self.encoder(128,128, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1),bias=False, batchnorm=True)

        ### bottleneck
        self.bt0 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True)
        self.bt1 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,2,2), padding=(0,2,2), bias=False, batchnorm=True)
        self.bt2 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,4,4), padding=(0,4,4), bias=False, batchnorm=True)

        ### decoder
        self.dc2 = self.decoder((1, 32, 32), 256 + 256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 32
        self.dc1 = self.decoder((1, 64, 64), 256 + 128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 64
        self.dc0 = self.decoder((1, 128, 128), 128 + 64, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=True, batchnorm=True) # 128
        
        ### predictors
        self.clip_diff_p = nn.Conv3d(64, 3, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        #self.activation = nn.Softshrink(0.025)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        if self.nl:
            e0 = self.enl0(self.ec0(x))
            e1 = self.enl1(self.ec1(e0)) # 64x(64) --> 128x(32)
        else:
            e0 = self.ec0(x)
            e1 = self.ec1(e0)
        e2 = self.ec2(e1) # 128x(32) --> 256x(16)

        ### temporal convs
        t0 = self.tc0(e0) #
        t1 = self.tc1(e1)

        ### bottleneck
        bt0 = self.bt0(e2)  # 256x(16)
        bt1 = self.bt1(bt0) # 256x(16)
        bt2 = self.bt2(bt1) # 256x(16)

        ### decoder
        d2 = self.dc2(torch.cat((bt2, e2),1)) # 16 --> 32
        d1 = self.dc1(torch.cat((d2, t1),1)) # 32 --> 64
        d0 = self.dc0(torch.cat((d1, t0),1))  # 64 --> 128

        if self.diff:
            clip_pred = self.clip_diff_p(d0)
            return torch.tanh(clip_pred)
        else:
            #clip_pred = x[:,:,4,:,:].unsqueeze(2) + self.activation(self.clip_diff_p(d0)) # 64 --> 3
            clip_pred = x[:,:,4,:,:].unsqueeze(2) + self.clip_diff_p(d0) # 64 --> 3
        # return clip_pred
        # return torch.tanh(clip_pred)*0.5
        # return torch.clamp(clip_pred, 0, 1)
        # return torch.clamp(clip_pred, -0.5, 0.5)
        return torch.sigmoid(clip_pred)

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
        self.ec0 = GatedConvolution(6, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False, type='2d')
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
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
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

        ### decoder
        d2 = self.dc2(torch.cat((bt2, e2),1)) 
        d1 = self.dc1(torch.cat((d2, t1), 1))
        d0 = self.dc0(torch.cat((d1, t0), 1))

        clip_pred = self.clip_diff_p(d0)

        return torch.sigmoid(clip_pred)

class ICNetDeepGate(nn.Module):
    def __init__(self, opt):
        super(ICNetDeepGate, self).__init__()
        self.opt = opt

        ### encoder
        self.ec0 = GatedConvolution(3, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.ec1 = GatedConvolution(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.ec2 = GatedConvolution(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1), bias=False)

        ### temporal convs
        self.tc0 = GatedConvolution(64, 64, kernel_size=(5,3,3), stride=(1,1,1), padding=(0,1,1),bias=False)
        self.tc1 = GatedConvolution(128,128, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1),bias=False)

        ### bottleneck
        self.bt0 = GatedConvolution(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,1,1), padding=(0,1,1), bias=False)
        self.bt1 = GatedConvolution(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,2,2), padding=(0,2,2), bias=False)
        self.bt2 = GatedConvolution(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,4,4), padding=(0,4,4), bias=False)

        ### decoder
        self.dc2 = GatedUpConvolution((1, 32, 32), 256 + 256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.dc1 = GatedUpConvolution((1, 64, 64), 256 + 128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.dc0 = GatedUpConvolution((1, 128, 128), 128 + 64, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)

        self.clip_diff_p = nn.Conv3d(64, 3, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=True)

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


class ICNetDeepGate2step(nn.Module):
    def __init__(self, opt):
        super(ICNetDeepGate2step, self).__init__()
        self.opt = opt

        self.coarse_network = ICNetDeepGate(self.opt)
        self.refine_network = ICNetDeepGate2D(self.opt)

    def forward(self, x):

        coarse_out = self.coarse_network(x)
        refine_in  = torch.cat((coarse_out.squeeze(), x[:,:,4,:,:]), 1)
        refine_out = self.refine_network(refine_in)

        return coarse_out, refine_out.unsqueeze(2)

class ICNetDeep(nn.Module):
    def __init__(self, opt):
        super(ICNetDeep, self).__init__()
        self.opt = opt
# =========================== NET D =============================
        ### encoder
        self.ec0_0 = self.encoder(3, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False, batchnorm=True) #64 
        self.ec0_1 = self.encoder(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True) #64 
        self.enl0 = NonLocalBlock(64) if opt.nl else None

        self.ec1_0 = self.encoder(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False, batchnorm=True)
        self.ec1_1 = self.encoder(128, 128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True) #32
        self.enl1 = NonLocalBlock(128) if opt.nl else None

        self.ec2_0 = self.encoder(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1), bias=False, batchnorm=True) #16
        self.ec2_1 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) #16
        self.enl2 = NonLocalBlock(256) if opt.nl else None

        ### temporal convs
        self.tc0_0 = self.encoder(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False, batchnorm=True)
        self.tc0_1 = self.encoder(64, 64, kernel_size=(5,3,3), stride=(1,1,1), padding=(0,1,1),bias=False, batchnorm=True)
        self.tnl0 = NonLocalBlock(64) if opt.nl else None

        self.tc1_0 = self.encoder(128,128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False, batchnorm=True)
        self.tc1_1 = self.encoder(128,128, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1),bias=False, batchnorm=True)
        self.tnl1 = NonLocalBlock(128) if opt.nl else None

        ### bottleneck
        self.bt0 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True)
        self.bt1 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,2,2), padding=(0,2,2), bias=False, batchnorm=True)
        self.bt2 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,4,4), padding=(0,4,4), bias=False, batchnorm=True)
        self.bnl = NonLocalBlock(256) if opt.nl else None

        ### decoder
        self.dc2_0 = self.decoder((1, 32, 32), 256 + 256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 32
        self.dc2_1 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 32
        self.dnl0 = NonLocalBlock(256) if opt.nl else None

        self.dc1_0 = self.decoder((1, 64, 64), 256 + 128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 64
        self.dc1_1 = self.encoder(128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 64
        self.dnl1 = NonLocalBlock(128) if opt.nl else None

        self.dc0 = self.decoder((1, 128, 128), 128 + 64, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=True, batchnorm=True) # 128

        
        ### predictors
        self.clip_diff_p = nn.Conv3d(64, 3, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=True)
        #self.activation = nn.Softshrink(0.025)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        e0_0 = self.ec0_0(x)
        e0_1 = self.ec0_1(e0_0)
        # print('e0_1',e0_1.size())

        e1_0 = self.ec1_0(e0_1)
        e1_1 = self.ec1_1(e1_0)
        # print('e1_1',e1_1.size())
        
        e2_0 = self.ec2_0(e1_1) # 128x(32) --> 256x(16)
        e2_1 = self.ec2_1(e2_0) # 128x(32) --> 256x(16)
        # print('e2_1',e2_1.size())

        ### temporal convs
        t0_0 = self.tc0_0(e0_1) #
        t0_1 = self.tc0_1(t0_0)
        # print('t0_1',t0_1.size())

        t1_0 = self.tc1_0(e1_1)
        t1_1 = self.tc1_1(t1_0)
        # print('t1_1',t1_1.size())

        ### bottleneck
        bt0 = self.bt0(e2_1)  # 256x(16)
        bt1 = self.bt1(bt0) # 256x(16)
        bt2 = self.bt2(bt1) # 256x(16)
        if self.opt.nl:
            bt2 = self.bnl(bt2)
        # print(bt2.size())

        ### decoder
        d2_0 = self.dc2_0(torch.cat((bt2, e2_1),1)) # 16 --> 32
        d2_1 = self.dc2_1(d2_0)
        if self.opt.nl:
            d2_1 = self.dnl0(d2_1)
        d1_0 = self.dc1_0(torch.cat((d2_1, t1_1),1)) # 32 --> 64
        d1_1 = self.dc1_1(d1_0)
        if self.opt.nl:
            d1_1 = self.dnl1(d1_1)
        d0 = self.dc0(torch.cat((d1_1,t0_1),1))

        clip_pred = x[:,:,4,:,:].unsqueeze(2) + self.clip_diff_p(d0) # 64 --> 3

        return torch.sigmoid(clip_pred)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ICNetResidual_DBI(nn.Module):
    def __init__(self, opt):
        super(ICNetResidual_DBI, self).__init__()

        # =========================== NET D =============================
        ### encoder
        self.ec0_0 = self.encoder(3, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False, batchnorm=True) #64 
        self.ec0_1 = self.encoder(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True) #64 

        self.ec1_0 = self.encoder(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False, batchnorm=True)
        self.ec1_1 = self.encoder(128, 128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False, batchnorm=True) #32

        self.ec2_0 = self.encoder(128, 256, kernel_size=(3,3,3), stride=(1,2,2), padding=(0,1,1), bias=False, batchnorm=True) #16
        self.ec2_1 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) #16

        ### temporal convs
        self.tc0_0 = self.encoder(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False, batchnorm=True)
        self.tc0_1 = self.encoder(64, 64, kernel_size=(5,3,3), stride=(1,1,1), padding=(0,1,1),bias=False, batchnorm=True)

        self.tc1_0 = self.encoder(128,128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1),bias=False, batchnorm=True)
        self.tc1_1 = self.encoder(128,128, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1),bias=False, batchnorm=True)

        ### bottleneck
        self.bt0 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True)
        self.bt1 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,2,2), padding=(0,2,2), bias=False, batchnorm=True)
        self.bt2 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,4,4), padding=(0,4,4), bias=False, batchnorm=True)

        ### decoder
        self.dc2_0 = self.decoder((1, 32, 32), 256 + 256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 32
        self.dc2_1 = self.encoder(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 32

        self.dc1_0 = self.decoder((1, 64, 64), 256 + 128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 64
        self.dc1_1 = self.encoder(128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, batchnorm=True) # 64

        self.dc0 = self.decoder((1, 128, 128), 128 + 64, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=True, batchnorm=True) # 128
    
        # =========================== NET E =============================

        layers_E = []
        layers_E.append(self.encoder(67,64,kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,1,1),padding=(0,1,1), bias = False))
        for i in range(3):
            layers_E.append(BasicBlock(64,64,kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias = False))
        layers_E.append( nn.Conv3d(64,3,kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=True))
        self.netE = nn.Sequential(*layers_E)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward_once(self, x):

        ### encoder
        e0_0 = self.ec0_0(x)
        e0_1 = self.ec0_1(e0_0)
        #print('e0_1',e0_1.size())
        e1_0 = self.ec1_0(e0_1)
        e1_1 = self.ec1_1(e1_0)
        #print('e1_1',e1_1.size())
        e2_0 = self.ec2_0(e1_1) # 128x(32) --> 256x(16)
        e2_1 = self.ec2_1(e2_0) # 128x(32) --> 256x(16)
        #print('e2_1',e2_1.size())
        ### temporal convs
        t0_0 = self.tc0_0(e0_1) #
        t0_1 = self.tc0_1(t0_0)
        #print('t0_1',t0_1.size())
        t1_0 = self.tc1_0(e1_1)
        t1_1 = self.tc1_1(t1_0)
        #print('t1_1',t1_1.size())

        ### bottleneck
        bt0 = self.bt0(e2_1)  # 256x(16)
        bt1 = self.bt1(bt0) # 256x(16)
        bt2 = self.bt2(bt1) # 256x(16)
        #print(bt2.size())

        ### decoder
        d2_0 = self.dc2_0(torch.cat((bt2, e2_1),1)) # 16 --> 32
        d2_1 = self.dc2_1(d2_0)
        d1_0 = self.dc1_0(torch.cat((d2_1, t1_1),1)) # 32 --> 64
        d1_1 = self.dc1_1(d1_0)
        d0 = self.dc0(torch.cat((d1_1,t0_1),1))
        #res = self.conv_last(d0)
        #print(d0.size())

        return d0


    def forward(self, x):
        res = self.forward_once(x)
        res2 = self.netE(torch.cat((res,x[:,:,4:5,:,:]),1)) # 64 + 3
        #return torch.clamp(x[:,:,4:5,:,:] + res2,0,1)
        return torch.clamp(x[:,:,4:5,:,:] + res2,-1,1)
        #return torch.sigmoid(x[:,:,4:5,:,:]+ res2)
        #return x[:,:,4:5,:,:] + res2
        #return x[:,:,4:5,:,:]+res2

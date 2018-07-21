import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, computation_compression=2, mode='embedded'):
        super(NonLocalBlock, self).__init__()
        assert ( mode == 'embedded' and computation_compression > 1 )
        self.mode = mode
        if mode not in ['gaussian', 'dot', 'embedded']:
            raise ValueError('mode must be one of gaussian, embedded, dot')
        self.channels = in_channels
        self.computation_compression = computation_compression
        if mode in ['dot', 'embedded']:
            self.theta = nn.Conv3d(in_channels, in_channels // 2, kernel_size=(1,1,1), bias=False)
            self.phi = nn.Conv3d(in_channels, in_channels // 2, kernel_size=(1,1,1), bias=False)
            if mode == 'embedded':
                self.pool1 = nn.MaxPool1d(kernel_size=computation_compression)
        self.g = nn.Conv3d(in_channels, in_channels // 2, kernel_size=(1,1,1), bias=False) 
        self.pool2 = nn.MaxPool1d(kernel_size=computation_compression)
        self.expand = nn.Conv3d(in_channels // 2, in_channels, kernel_size=(1,1,1), bias=False)
        self.bn = nn.BatchNorm3d(in_channels)
   
    def forward(self, x):
        x_size = x.size()
        batchsize, channels, dim1, dim2, dim3 = x_size
        if self.mode == 'gaussian':
            x_theta = x.view(batchsize, self.channels, -1).permute(0,2,1)
            x_phi = x.view(batchsize, self.channels, -1)
            f = torch.bmm(x_theta, x_phi)
            f = F.softmax(f.permute(2,1,0)).permute(2,1,0)
        elif self.mode == 'dot':
            x_theta = self.theta(x)
            x_theta = x_theta.view(batchsize, channels // 2, -1).permute(0,2,1)
            x_phi = self.phi(x)
            x_phi = x_phi.view(batchsize, channels // 2, -1)
            f = torch.bmm(x_theta, x_phi)
            if batchsize is not None:
                f = (lambda z: 1./ batchsize * z)(f)
            else:
                f = (lambda z: 1./ 128 * z)(f)
        else:
            x_theta = self.theta(x)
            x_theta = x_theta.view(batchsize, channels // 2, -1).permute(0,2,1)
            x_phi = self.phi(x)
            x_phi = x_phi.view(batchsize, channels // 2, -1)
            if self.computation_compression > 1:
                x_phi = self.pool1(x_phi)
            f = torch.bmm(x_theta, x_phi)
            f = F.softmax(f.permute(2,1,0)).permute(2,1,0)

        x_g = self.g(x)
        x_g = x_g.view(batchsize, channels // 2, -1)
        if self.computation_compression > 1:
            x_g = self.pool2(x_g)

        # expand filters
        x_g = x_g.permute(0,2,1)
        y = torch.bmm(f, x_g)
        y = y.permute(0,2,1).contiguous().view(batchsize, channels // 2, dim1, dim2, dim3)
        y = self.expand(y)
        y = self.bn(y)

        # residual connection
        output = torch.add(x, y)

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb

class Discriminator(nn.Module):
    def __init__(self, opt=None):
        super(Discriminator, self).__init__()
        self.nc = 32
        if opt is not None:
            stride_v1 = (1,2,2)
            stride_v2 = (2,2,2)
            stride =(stride_v1, stride_v2)[opt.sample_duration==16]
        else:
            stride = (1,2,2)

        self.main = nn.Sequential(
            # 3x128x128
            nn.Conv3d(3, self.nc, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(self.nc, self.nc * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(self.nc * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(self.nc * 2, self.nc * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(self.nc * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(self.nc * 4, self.nc * 8, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(self.nc * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(self.nc * 8, self.nc * 8, kernel_size=3, stride=(1,2,2), padding=1, bias=False),
            nn.BatchNorm3d(self.nc * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(self.nc * 8, 1, kernel_size=(1,4,4), stride=(1,2,2), padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)

        return output.view(-1, 1)



class Discriminator2D(nn.Module):
    def __init__(self, opt=None):
        super(Discriminator2D, self).__init__()


        self.main = nn.Sequential(
            # (3+3)x1x128x128
            nn.Conv3d(6, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,2,2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,2,2)),
            nn.BatchNorm3d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, kernel_size=(1,4,4), stride=(1,1,1), padding=(0,2,2)),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.main(x)



if __name__=='__main__':
    net = Discriminator()
    x = Variable(torch.ones((1,3,8,128,128)))

    output = net(x)
    print(output.size())
    pdb.set_trace()

    

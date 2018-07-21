import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pdb
from torch.autograd import Variable

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(
                        vgg16.features[0],
                        vgg16.features[1],
                        vgg16.features[2],
                        vgg16.features[3],
                        vgg16.features[4],
                        )
        self.enc_2 = nn.Sequential(
                        vgg16.features[5],
                        vgg16.features[6],
                        vgg16.features[7],
                        vgg16.features[8],
                        vgg16.features[9],
                        )
        self.enc_3 = nn.Sequential(
                        vgg16.features[10],
                        vgg16.features[11],
                        vgg16.features[12],
                        vgg16.features[13],
                        vgg16.features[14],
                        vgg16.features[15],
                        vgg16.features[16],
                        )

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]



if __name__=='__main__':
    net = VGG16FeatureExtractor()

    x = Variable(torch.ones((8,3,128,128)))

    output = net(x)
    pdb.set_trace()

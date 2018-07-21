import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from models.discriminator import Discriminator
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, ZeroOneNormalize, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop, MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, ColorJitter, PerImageNormalize, RandomSampleCrop)
from temporal_transforms import LoopPadding, TemporalRandomCrop, RandomTemporalFlip, TemporalRandomCropMirror
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
import pdb

from loss import InpaintingLoss
from vgg16 import VGG16FeatureExtractor
from augmentations import PhotometricDistort
from networks import define_G, define_D, GANLoss

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        # opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        #opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            # opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
            tmp_root_path = '/ssd2/vid_inpaint/Track2/starting_kit_tmp'
            opt.resume_path = os.path.join(tmp_root_path, opt.resume_path)
        #if opt.pretrain_path:
            #opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    
    # **************  VISDOM  ****************
    global viz, train_lot, test_lot
    if opt.visdom:
        import visdom
        viz = visdom.Visdom()
        if not opt.no_train:
            if opt.is_AE:
                if opt.grad or opt.ssim:
                    if opt.two_step:
                        train_lot = viz.line(X = torch.zeros((1,7)).cpu(), Y = torch.zeros((1,7)).cpu(),opts = dict(xlabel='Epoch',ylabel='Loss&Acc',title='%s Train'%opt.prefix, legend=['loss_img*50','loss_grad*1000','dssim*100','psnr1', 'mse1*1000','psnr2', 'mse2*1000']))        
                    else:
                        train_lot = viz.line(X = torch.zeros((1,5)).cpu(), Y = torch.zeros((1,5)).cpu(),opts = dict(xlabel='Epoch',ylabel='Loss&Acc',title='%s Train'%opt.prefix, legend=['loss_img*1000','loss_grad*1000','dssim*100','psnr', 'mse*1000']))
                else:
                    train_lot = viz.line(X = torch.zeros((1,3)).cpu(), Y = torch.zeros((1,3)).cpu(),opts = dict(xlabel='Epoch',ylabel='Loss&Acc',title='%s Train'%opt.prefix, legend=['loss_img*100','psnr', 'mse*100']))
        if not opt.no_val:
            val_lot = viz.line(X=torch.zeros((1,2)).cpu(),  Y=torch.zeros((1,2)).cpu(), opts=dict( xlabel='Epoch', ylabel='Loss&Acc', title='%s Val'%opt.prefix, legend=['loss', 'acc']))
    else:
        viz = None
        train_lot = None
        val_lot = None
    # ****************************************

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, _ = generate_model(opt)
    print(model)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    if opt.is_AE:
        if opt.wt_l2 == 1:
            criterion = nn.MSELoss()
            print('USING L2 LOSS')
        if opt.wt_l1 == 1:
            criterion = nn.L1Loss()
            print('USING L1 LOSS')
        if opt.grad:
            criterion2 = nn.L1Loss()
        elif opt.use_gan:
            criterion2 = nn.BCELoss()
        else:
            criterion2=None

        if opt.ssim:
            import pytorch_ssim
            criterion3 = pytorch_ssim.SSIM(window_size=3)
        else:
            criterion3 = None
    else:
        criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
        if criterion2 is not None:
            criterion2 = criterion2.cuda()
        if criterion3 is not None:
            criterion3 = criterion3.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    
    # ===================================================================================
    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center', 'custom']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c']) 
        elif opt.train_crop == 'custom':
            crop_method = RandomSampleCrop(opt.sample_size)
        clip_transform = None
        
        spatial_transform = Compose([ToTensor(opt.norm_value), ColorJitter(0.05, 0.05), norm_method])
        temporal_transform = TemporalRandomCrop(int(opt.sample_duration*opt.t_stride))

        training_data = get_training_set(opt, spatial_transform, temporal_transform)

        train_loader = torch.utils.data.DataLoader( training_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads, pin_memory=True)            
        train_logger = Logger( os.path.join(opt.result_path, 'train.log'), ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger( os.path.join(opt.result_path, 'train_batch.log'), ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
        if opt.two_step:
            optimizer = None
            netG = define_G(3, 3, 64, 'batch', False)

            if opt.end:
                optimizerG = optim.Adam(list(model.parameters())+list(netG.parameters()),opt.learning_rate, betas=(0.9,0.999))
            else:
                optimizerG = optim.Adam(netG.parameters(),opt.learning_rate, betas=(0.9, 0.999))
            print(netG)
            if opt.use_gan:
                netD = define_D(3+3, 64, 'batch', use_sigmoid=False)
                optimizerD = optim.Adam(netD.parameters(), opt.learning_rate, betas=(0.9, 0.999))
                print(netD)
            else:
                netD = None
                optimizerD=None
        else:
            if opt.use_gan:
                netD = define_D(3+3, 64, 'batch', use_sigmoid=False)
                optimizerD = optim.Adam(netD.parameters(), opt.learning_rate, betas=(0.9, 0.999))
                print(netD)
            else:
                netD = None
                optimizerD=None
            optimizerG = None       
            netG=None

    # ===================================================================================
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)

        mask_transform = Compose([ToTensor(opt.norm_value), ZeroOneNormalize()])
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, mask_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_loss, train_acc = train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger, viz, train_lot,netD, optimizerD, criterion2, netG, optimizerG, criterion3)

        if not opt.no_val:
            validation_loss, validation_acc = val_epoch(i, val_loader, model, criterion, opt,  val_logger)
            if opt.visdom:
                viz.line(X=torch.ones((1,2)).cpu()*(i-1), Y=torch.Tensor( [[validation_loss, validation_acc*10]]), win=val_lot, update='append')

    # ===================================================================================

    if opt.test:

        spatial_transform = Compose([ToTensor(opt.norm_value), norm_method])
        temporal_transform = LoopPadding(opt.sample_duration)

        test_data = get_test_set(opt, spatial_transform=spatial_transform, temporal_transform=temporal_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        if opt.is_AE:
            if opt.two_step:
                netG = define_G(3, 3, 64, 'batch', False)
                if opt.pretrain_path:
                    print('netG_loading pretrained model {}'.format(opt.pretrain_path))
                    pretrain = torch.load(opt.pretrain_path)
                    print('netG_loading from', pretrain['arch'])
                    child_dict = netG.state_dict()
                    parent_list = pretrain['state_dict_2'].keys()
                    print('netG_Not loaded :')
                    parent_dict = {}
                    for c,_ in child_dict.items():
                        if c in parent_list:
                            parent_dict[c] = pretrain['state_dict_2'][c]
                        else:
                            print(c)
                    print('netG_length :',len(parent_dict.keys()))
                    child_dict.update(parent_dict)
                test.test_AE(test_loader, model, opt, netG)
            else:
                test.test_AE(test_loader, model, opt)
        else:
            test.test(test_loader, model, opt)


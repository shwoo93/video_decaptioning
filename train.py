import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import sys
import pdb
import matplotlib.pyplot as plt
import numpy as np
from utils import AverageMeter, calculate_accuracy, tensor2img, DxDy
import scoring
from loss import pixel_bce_with_logits
from scipy.misc import imsave
from PIL import Image, ImageDraw

import pytorch_ssim
from pytorch_misc import clip_grad_norm

# 
LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, viz, train_lot, netD=None, optimizerD=None,
                criterion2=None, netG=None, optimizerG=None, criterion3=None):
    print('train at epoch {}'.format(epoch))
    model.train()
    if opt.two_step:
        netG.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    if 'icnet' in opt.model:
        losses_img = AverageMeter()
        losses_img_ = AverageMeter()
    else:
        losses_text = AverageMeter()
        losses_non_text = AverageMeter()
        losses_mask = AverageMeter()
    losses_frames = AverageMeter()

    losses_ssim = AverageMeter()
    losses_ssim_ = AverageMeter()

    losses_grad = AverageMeter()
    losses_grad_ = AverageMeter()


    losses_d = AverageMeter()
    losses_g_gan = AverageMeter()

    mses1 = AverageMeter()
    psnrs1 = AverageMeter()
    mses2 = AverageMeter()
    psnrs2 = AverageMeter()

    end_time = time.time()
    j=0

    for i, (inputs, targets) in enumerate(data_loader):

        data_time.update(time.time() - end_time)
        if opt.diff:
            targets = targets - inputs[:,:,4:5,:,:]

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
            targets = Variable(targets)
            inputs = inputs.cuda(async=True)
            if opt.two_step:
                netG.cuda()
            if opt.use_gan:
                netD.cuda()


        inputs = Variable(inputs)
        bs = inputs.size(0)
        outputs = model(inputs)
        

        # if (i % 50) == 0:
        #     save_path = os.path.join('/ssd2/vid_inpaint/Track2/starting_kit_tmp/results', opt.prefix, 'process', str(epoch))
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     if opt.t_shrink:
        #         idx = (opt.sample_duration)//2
        #     else:
        #         idx=0
        #     in_img = tensor2img(inputs[0,:,idx,:,].data.cpu().numpy(), opt)
        #     imsave(os.path.join(save_path, 'in_img_%04d.jpg'%(i)), in_img)
        #     out_img = tensor2img(outputs[0,:,0,:,].data.cpu().numpy(), opt)
        #     imsave(os.path.join(save_path, 'out_img_%04d.jpg'%(i)), out_img)


        if 'icnet' in opt.model:

            if not opt.two_step:
                if len(outputs) == 2:
                    loss_img_coarse = criterion(outputs[0], targets)
                    loss_img_refine = criterion(outputs[1], targets)
                    
                    losses_img.update(loss_img_coarse.data[0],bs)
                    losses_img_.update(loss_img_refine.data[0],bs)

                    loss_coarse = loss_img_coarse
                    loss_refine = loss_img_refine
                else:
                    loss_img = criterion(outputs, targets)
                    losses_img.update(loss_img.data[0],bs)
                    loss = loss_img

                if opt.minl1:
                    loss_minl1 = torch.abs(outputs -targets)
                    loss_minl1 = F.adaptive_max_pool3d(loss_minl1,1).mean()
                    loss = loss + loss_minl1

                if opt.grad:
                    if len(outputs) == 2:
                        dxo_coarse, dyo_coarse = DxDy(outputs[0])
                        dxo_refine, dyo_refine = DxDy(outputs[1])
                        dxt, dyt = DxDy(targets)

                        loss_grad_x_coarse = criterion2(dxo_coarse,dxt)
                        loss_grad_y_coarse = criterion2(dyo_coarse,dyt)
                        
                        loss_grad_x_refine = criterion2(dxo_refine,dxt)
                        loss_grad_y_refine = criterion2(dyo_refine,dyt)

                        losses_grad.update(loss_grad_x_coarse.data[0]+loss_grad_y_coarse.data[0], bs)
                        losses_grad_.update(loss_grad_x_refine.data[0]+loss_grad_y_refine.data[0], bs)

                        loss_coarse = loss_coarse + loss_grad_x_coarse + loss_grad_y_coarse
                        loss_refine = loss_refine + loss_grad_x_refine + loss_grad_y_refine
                    else:

                        dxo, dyo = DxDy(outputs)
                        dxt, dyt = DxDy(targets)
                        loss_grad_x = criterion2(dxo,dxt)
                        loss_grad_y = criterion2(dyo,dyt)
                        losses_grad.update(loss_grad_x.data[0]+loss_grad_y.data[0], bs)
                        loss = loss + loss_grad_x+loss_grad_y

                if opt.ssim:
                    if len(outputs) == 2:
                        loss_ssim_coarse = -criterion3(outputs[0].squeeze(), targets.squeeze())
                        loss_ssim_refine = -criterion3(outputs[1].squeeze(), targets.squeeze())

                        losses_ssim.update(loss_ssim_coarse.data[0]*(-1),bs)
                        losses_ssim_.update(loss_ssim_refine.data[0]*(-1),bs)

                        loss_coarse = loss_coarse + loss_ssim_coarse
                        loss_refine = loss_refine + loss_ssim_refine
                    else:
                        loss_ssim = -criterion3(outputs.squeeze(), targets.squeeze())
                        losses_ssim.update(loss_ssim.data[0]*(-1),bs)
                        loss = loss + loss_ssim

                if opt.use_gan:
                    optimizerD.zero_grad()
                    # train with fake
                    fake_ab = torch.cat((inputs[:,:,4:5,:,:], outputs), 1)
                    pred_fake = netD.forward(fake_ab.detach()) #128x1x1x67x67
                    label_fake = Variable(torch.FloatTensor(pred_fake.size())).cuda()
                    label_fake.data.resize_(pred_fake.size()).fill_(0)
                    if opt.mingan:
                        loss_d_fake = pixel_bce_with_logits(pred_fake, label_fake)
                        loss_d_fake = F.adaptive_max_pool3d(loss_d_fake,1).mean()
                    else:
                        loss_d_fake = F.binary_cross_entropy_with_logits(pred_fake, label_fake)
                    real_ab = torch.cat((inputs[:,:,4:5,:,:], targets), 1)
                    pred_real = netD.forward(real_ab)
                    label_real=Variable(torch.FloatTensor(pred_fake.size())).cuda()
                    label_real.data.resize_(pred_real.size()).fill_(1)
                    if opt.mingan:
                        loss_d_real = pixel_bce_with_logits(pred_real, label_real)
                        loss_d_real = F.adaptive_max_pool3d(loss_d_real,1).mean()
                    else:
                        loss_d_real = F.binary_cross_entropy_with_logits(pred_real, label_real)
                    # Combined loss
                    loss_d = (loss_d_fake + loss_d_real) * 0.5
                    loss_d.backward()
                    optimizerD.step()
                    ############################
                    # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
                    ##########################
                    optimizer.zero_grad()
                    # First, G(A) should fake the discriminator
                    fake_ab = torch.cat((inputs[:,:,4:5,:,:], outputs), 1)
                    pred_fake = netD.forward(fake_ab)
                    label_real=Variable(torch.FloatTensor(pred_fake.size())).cuda()
                    label_real.data.resize_(pred_real.size()).fill_(1)
                    if opt.mingan:
                        loss_g_gan = pixel_bce_with_logits(pred_fake, label_real)
                        loss_g_gan = F.adaptive_max_pool3d(loss_g_gan,1).mean()
                    else:
                        loss_g_gan = F.binary_cross_entropy_with_logits(pred_fake, label_real)
                    losses_g_gan.update(loss_g_gan.data[0],bs)
                    loss = loss + loss_g_gan * 0.01



            elif opt.two_step:
                outputs2 = netG(torch.cat((inputs[:,:,4:5,:,:],outputs),1))

                if opt.minl1:
                    loss_g = torch.abs(outputs2 -targets)
                    loss_g = F.adaptive_max_pool3d(loss_g,1).mean() *10
                else:
                    loss_g = criterion(outputs2,targets) * 10
                losses_img.update(loss_g.data[0],bs)

                if opt.use_gan:
                    optimizerD.zero_grad()
                    # train with fake
                    fake_ab = torch.cat((outputs.detach(), outputs2), 1)
                    pred_fake = netD.forward(fake_ab.detach()) #128x1x1x67x67
                    label_fake = Variable(torch.FloatTensor(pred_fake.size())).cuda()
                    label_fake.data.resize_(pred_fake.size()).fill_(0)
                    #loss_d_fake = F.binary_cross_entropy_with_logits(pred_fake.squeeze(), label_fake.squeeze())
                    if opt.mingan:
                        loss_d_fake = pixel_bce_with_logits(pred_fake, label_fake)
                        loss_d_fake = F.adaptive_max_pool3d(loss_d_fake,1).mean()
                    else:
                        loss_d_fake = F.binary_cross_entropy_with_logits(pred_fake, label_fake)

                    #loss_d_fake = F.binary_cross_entropy(pred_fake, label)
                    # train with real
                    real_ab = torch.cat((outputs.detach(), targets), 1)
                    pred_real = netD.forward(real_ab)
                    label_real=Variable(torch.FloatTensor(pred_fake.size())).cuda()
                    label_real.data.resize_(pred_real.size()).fill_(1)
                    #loss_d_real = F.binary_cross_entropy_with_logits(pred_real.squeeze(), label_real.squeeze())
                    if opt.mingan:
                        loss_d_real = pixel_bce_with_logits(pred_real, label_real)
                        loss_d_real = F.adaptive_max_pool3d(loss_d_real,1).mean()
                    else:
                        loss_d_real = F.binary_cross_entropy_with_logits(pred_real, label_real)


                    # Combined loss
                    loss_d = (loss_d_fake + loss_d_real) * 0.5
                    loss_d.backward()
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
                    ##########################
                    optimizerG.zero_grad()
                    # First, G(A) should fake the discriminator
                    fake_ab = torch.cat((outputs, outputs2), 1)
                    pred_fake = netD.forward(fake_ab)
                    label_real=Variable(torch.FloatTensor(pred_fake.size())).cuda()
                    label_real.data.resize_(pred_real.size()).fill_(1)
                    #loss_g_gan = F.binary_cross_entropy_with_logits(pred_fake.squeeze(), label_real.squeeze())
                    if opt.mingan:
                        loss_g_gan = pixel_bce_with_logits(pred_fake, label_real)
                        loss_g_gan = F.adaptive_max_pool3d(loss_g_gan,1).mean()
                    else:
                        loss_g_gan = F.binary_cross_entropy_with_logits(pred_fake, label_real)

                    losses_g_gan.update(loss_g_gan.data[0],bs)

                    loss_g = loss_g + loss_g_gan

                if opt.grad:
                    dxo, dyo = DxDy(outputs)
                    dxt, dyt = DxDy(targets)
                    if opt.mingrad:
                        loss_grad_x = torch.abs(dxo-dxt)
                        loss_grad_x = F.adaptive_max_pool3d(loss_grad_x,1).mean()
                        loss_grad_y = torch.abs(dyo-dyt)
                        loss_grad_y = F.adaptive_max_pool3d(loss_grad_y,1).mean()
                    else:
                        loss_grad_x = criterion2(dxo,dxt)
                        loss_grad_y = criterion2(dyo,dyt)
                    losses_grad.update(loss_grad_x.data[0]+loss_grad_y.data[0], bs)
                    loss_g = loss_g + loss_grad_x+loss_grad_y

                if opt.ssim:
                    loss_ssim_g = -criterion3(outputs2.squeeze(), targets.squeeze())
                    losses_ssim.update(loss_ssim_g.data[0]*(-1),bs)
                    loss_g = loss_g + loss_ssim_g

                loss_g.backward()
                optimizerG.step()

        else:
            print("Not Implemented Error")

        score = {}
        if 'icnet' in opt.model:
            if len(outputs) == 2:
                outs=[outputs[0], outputs[1]]
            else:
                outs=[outputs]
            if opt.two_step:
                outs.append(outputs2)
        else:
            pass
        for idx, out in enumerate(outs):
            n_batch = out.size(0)
            n_frame = out.size(2)
            for meth in [scoring.PSNR, scoring.MSE]: #scoring.DSSIM
                name = meth.__name__
                results = []
                res = 0.
                for batch in range(n_batch):
                    out_ = out[batch,:,:,:,:]
                    target = targets[batch,:,:,:,:]
                    if opt.t_shrink:
                        res = meth(out_.data.cpu().numpy(),target.data.cpu().numpy())

                    else:
                        pass
                        res /= float(n_frame)
                    results.append(res)
                mres = np.mean(results)
                score[name]=mres # score['PSNR'] score['MSE']

            if idx == 0:
                if score['MSE'] == float('nan'):
                    pdb.set_Trace()
                else:
                    mses1.update(score['MSE'])
                if score['PSNR'] != float('Inf'):
                    psnrs1.update(score['PSNR'])
            elif idx ==1:
                if score['MSE'] == float('nan'):
                    pdb.set_Trace()
                else:
                    mses2.update(score['MSE'])
                if score['PSNR'] != float('Inf'):
                    psnrs2.update(score['PSNR'])

        if not opt.two_step:
            if not opt.use_gan:
                optimizer.zero_grad()

            if len(outputs) == 2:
                loss = loss_coarse + loss_refine
            # print('loss_coarse: {} loss_img_coarse: {} loss_grad_x_coarse: {} loss_grad_y_coarse: {} loss_ssim_coarse: {}'.format(
            #         loss_coarse, loss_img_coarse, loss_grad_x_coarse, loss_grad_y_coarse, loss_ssim_coarse
            #         ))
            # print('loss_refine: {} loss_img_refine: {} loss_grad_x_refine: {} loss_grad_y_refine: {} loss_ssim_refine: {}'.format(
            #         loss_refine, loss_img_refine, loss_grad_x_refine, loss_grad_y_refine, loss_ssim_refine
            #         ))
            loss.backward()

            total_norm = clip_grad_norm(
            [(n, p) for n, p in model.named_parameters() if p.grad is not None],
            max_norm=1., verbose=True, clip=True)
            if total_norm == float('nan'):
                pdb.set_trace()
            optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()


        # --------------------------------------------------------------------------

        if opt.is_AE:
            if 'icnet' in opt.model:
                if opt.two_step:
                    print('Epoch: [{0}][{1}/{2}]\t'  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'   'Loss_img {loss_img.val:.3f} ({loss_img.avg:.3f})\t' 'PSNR1 {psnr1.val:.3f} ({psnr1.avg:.3f})\t' 'MSE1 {mse1.val:.5f} ({mse1.avg:.5f})\t' 'PSNR2 {psnr2.val:.3f} ({psnr2.avg:.3f})\t' 'MSE2 {mse2.val:.5f} ({mse2.avg:.5f})\t' 'G_GAN {loss_grad.val:.5f} ({loss_grad.avg:.5f})\t' 'DSSIM {dssim_val:.5f} ({dssim_avg:.5f})\t'.format( epoch,  i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss_img =losses_img, psnr1=psnrs1, mse1=mses1, psnr2=psnrs2, mse2=mses2, loss_grad=losses_g_gan, dssim_val=(1.0-losses_ssim.val)*0.5, dssim_avg=(1.0-losses_ssim.avg)*0.5))           

                else:
                    if len(outputs) == 2:
                        print('Epoch: [{0}][{1}/{2}]\t'  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 'Loss_img_coarse {loss_img.val:.3f} ({loss_img.avg:.3f})\t' 'Loss_img_refine {loss_img_.val:.3f} ({loss_img_.avg:.3f})\t' 'PSNR1 {psnr1.val:.3f} ({psnr1.avg:.3f})\t' 'MSE1 {mse1.val:.5f} ({mse1.avg:.5f})\t' 'PSNR2 {psnr2.val:.3f} ({psnr2.avg:.3f})\t' 'MSE2 {mse2.val:.5f} ({mse2.avg:.5f})\t' 'Grad_coarse {loss_grad.val:.5f} ({loss_grad.avg:.5f})\t' 'Grad_refine {loss_grad_.val:.5f} ({loss_grad_.avg:.5f})\t' 'DSSIM_coarse {dssim_val:.5f} ({dssim_avg:.5f})\t' 'DSSIM_refine {dssim_val_:.5f} ({dssim_avg_:.5f})\t'.format( epoch,  i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss_img =losses_img, loss_img_=losses_img_, psnr1=psnrs1, mse1=mses1, psnr2=psnrs2, mse2=mses2, loss_grad=losses_grad, loss_grad_=losses_grad_, dssim_val=(1.0-losses_ssim.val)*0.5, dssim_avg=(1.0-losses_ssim.avg)*0.5, dssim_val_=(1.0-losses_ssim_.val)*0.5, dssim_avg_=(1.0-losses_ssim_.avg)*0.5))
                    else:
                        print('Epoch: [{0}][{1}/{2}]\t'  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'   'Loss_img {loss_img.val:.3f} ({loss_img.avg:.3f})\t' 'PSNR1 {psnr1.val:.3f} ({psnr1.avg:.3f})\t' 'MSE1 {mse1.val:.5f} ({mse1.avg:.5f})\t' 'G_GAN {loss_grad.val:.5f} ({loss_grad.avg:.5f})\t' 'DSSIM {dssim_val:.5f} ({dssim_avg:.5f})\t'.format( epoch,  i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss_img =losses_img, psnr1=psnrs1, mse1=mses1, loss_grad=losses_grad, dssim_val=(1.0-losses_ssim.val)*0.5, dssim_avg=(1.0-losses_ssim.avg)*0.5))           

        else:
            pass
        
        if opt.visdom:                
            if (i+1)%(int(len(data_loader)//10)) == 0:
                if opt.grad or opt.ssim:
                    dssim = (1.0-losses_ssim.avg)*0.5
                    if opt.two_step:
                        viz.line(X=torch.ones((1,7)).cpu()*(j+10*(epoch-1)), Y=torch.Tensor( [[losses_img.avg*50, losses_grad.avg*100, dssim*100, psnrs1.avg, mses1.avg*1000,psnrs2.avg, mses2.avg*1000]]), win=train_lot, update='append')
                    else:
                       viz.line(X=torch.ones((1,5)).cpu()*(j+10*(epoch-1)), Y=torch.Tensor( [[losses_img.avg*1000, losses_grad.avg*1000, dssim*100, psnrs1.avg, mses1.avg*1000]]), win=train_lot, update='append')
                else:
                    viz.line(X=torch.ones((1,3)).cpu()*(j+10*(epoch-1)), Y=torch.Tensor( [[losses_img.avg*100, psnrs.avg, mses.avg*100]]), win=train_lot, update='append')
                j+=1  

    # --------------------------------------------------------------------------

    if epoch % opt.checkpoint == 0:
        if not opt.two_step:
            save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
        else:
            save_file_path = os.path.join(opt.result_path, 'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'arch': opt.arch,
                'state_dict_1': model.state_dict(),
                'state_dict_2': netG.state_dict(),
                'optimizer': optimizerG.state_dict(),
            }
            torch.save(states, save_file_path)

    '''if 'icnet' in opt.model:
        losses = losses_img.avg + errG_Ds.avg
    else:
        losses = losses_text.avg + losses_non_text.avg + losses_mask.avg + errG_Ds.avg'''

    return losses_img.avg, losses_img.avg
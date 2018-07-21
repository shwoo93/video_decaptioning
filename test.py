import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json
from skimage.measure import compare_ssim as ssim,compare_mse,compare_psnr
from utils import AverageMeter, tensor2img, DxDy
from data_manager import createVideoClip
from poissonblending import blend
import numpy as np
import cv2
import pdb
from scipy.misc import imsave

def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i]],
            'score': sorted_scores[i]
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        #print(i)
        #continue
        #pdb.set_trace()
        data_time.update(time.time() - end_time)

        inputs = Variable(inputs, volatile=True)
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)


def test_AE(data_loader, model, opt, netG=None, model_=None):
    print('test_AE')
    import matplotlib.pyplot as plt
    if netG is not None:
        netG.cuda()
        netG.eval()
    
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mse_losses = AverageMeter()
    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}

    # folder = os.path.join(opt.root_path, 'dev', 'Y')
    folder = opt.result_path
    if not os.path.exists(folder):
        os.makedirs(fodler)

    ori_clips = []
    pred_clips = []
    masks = []
    clips = []
    for i, (inputs, path) in enumerate(data_loader):
        #print(i)

        name = 'Y' + path[0].split('/')[-1][1:] + '.mp4'
        if os.path.exists(os.path.join(folder,name)):
            print(name)
            continue
            
        data_time.update(time.time() - end_time)
        inputs = Variable(inputs, volatile=True)
        outputs = model(inputs)

        if netG is not None:
            outputs = netG(outputs)

        inputs = inputs[0,:,4,:,:].cpu().data.numpy()
        outputs= outputs[0,:,0,:,:].cpu().data.numpy()

        if opt.cut:
            diff = outputs - inputs
            tmp = (diff<0.01) * (diff>-0.01)
            #mu = tmp.mean()
            #outputs = outputs-mu
            outputs[tmp] = inputs[tmp]
   
        mse_losses.update(0)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})\t'.format( i + 1,len(data_loader), batch_time=batch_time, data_time=data_time, mse_losses=mse_losses))

        clips.append(outputs) # 1x3x1x128x128
        if opt.t_shrink:
            if (i+1) % 125 == 0: # last
                clips = [tensor2img(clip, opt) for clip in clips]
                final_clip = np.stack(clips)
                name = 'Y' + path[0].split('/')[-1][1:] + '.mp4'
                createVideoClip(final_clip, folder, name)
                clips = []
                print('Predicted video clip {} saving'.format(name))   
     
        else:
            print('Not Implemented Error')

    print('mse_losses:', mse_losses.avg)
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)


def normalize(x):
    return (x-x.min())/(x.max()-x.min())
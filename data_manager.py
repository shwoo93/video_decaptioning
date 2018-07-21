#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from PIL import Image
import numpy as np
import pims
import subprocess as sp
import re
import os, sys
import pickle
import pdb

fsize = 128 # 256

#root_dataset = '../dataset-mp4/' # Download from competition url
# root_dataset = '../dataset-sample/'
root_dataset = '/data1/vid_inpaint/Track2/dataset'
'''
data generator used for baseline1
load video clip and randomly choose 2 frames for training
'''
def generate_data(max_samples, batchsize, part): #part = train|dev|test
    while 1:
        samples = list(range(0, max_samples, batchsize))
        #np.random.shuffle(samples)
        for i in samples:
            X = []
            Y = []

            #Read a batch of clips from files
            j = 0
            while len(X) < batchsize:
                if part == 'train':
                    idxs = list(range(25*5))
                    np.random.shuffle(idxs)
                    idxs = idxs[:2] # keep only 2 random frames per clip on train mode
                else:
                    idxs = [50, 100] # only evaluate frames 50 and 100 on eval mode
           
                ok = True
                path = root_dataset+'/'+part+'/X/X'+str(i+j)+'.mp4'
                # pdb.set_trace()

                try:
                    Xj = pims.Video(root_dataset+'/'+part+'/X/X'+str(i+j)+'.mp4')[idxs]
                    Xj = np.array(Xj, dtype='float32') / 255.
                    Yj = pims.Video(root_dataset+'/'+part+'/Y/Y'+str(i+j)+'.mp4')[idxs]
                    Yj = np.array(Yj, dtype='float32') / 255.
                except:
                    print('Error clip number '+ str(i+j) + ' at  '+root_dataset+'/train/X/X'+str(i+j)+'.mp4'+ ' OR '+root_dataset+'/train/Y/Y'+str(i+j)+'.mp4')
                    ok = False
                    if i+j >= max_samples: j = 0
                if ok:
                    X.append(Xj)
                    Y.append(Yj)
                j = j + 1

            # make numpy and reshape
            X = np.asarray(X)
            X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
            Y = np.asarray(Y)
            Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
            yield (X, Y)

# return all frames from video clip
# returned frames are normalized
def getAllFrames(clipname):
    print(clipname)

    # open one video clip sample
    try:
        data = pims.Video(root_dataset+'/'+clipname)
    except:
        data = pims.Video(clipname)

    data = np.array(data, dtype='float32')
    length = data.shape[0]

    return data[:125] / 255.
        
# create video clip using 'ffmpeg' command
# clip: input data, supposed normalized (between 0 and 1)
# name: basename of output file
def createVideoClip(clip, folder, name):
    #clip = clip * 255.
    #clip = clip.astype('uint8')

    # write video stream #
    command = [ 'ffmpeg',
    '-y',  # overwrite output file if it exists
    '-f', 'rawvideo',
    '-s', '128x128', #'256x256', # size of one frame
    '-pix_fmt', 'rgb24',
    '-r', '25', # frames per second
    '-an',  # Tells FFMPEG not to expect any audio
    '-i', '-',  # The input comes from a pipe
    '-vcodec', 'libx264',
    '-b:v', '100k',
    '-vframes', '125', # 5*25
    '-s', '128x128', #'256x256', # size of one frame
    folder+'/'+name ]

    pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
    out, err = pipe.communicate(clip.tostring())
    pipe.wait()
    pipe.terminate()
    print(err)

################################### baseline2 #################################

# for baseline2, we precompute mini batches.
# don't need a generator since inputs are small dimension (patches)
def build_and_save_batches(max_samples, batchsize): #part = train|dev|test
        different_clips_per_batch = 10
        number_of_frames_per_clips = 2
        
        samples = list(range(max_samples))
        np.random.shuffle(samples)
        num_batch = 0
        for i in range(0, max_samples, different_clips_per_batch):
            X = []
            Y = []

            #Read a batch of clips from files
            j = 0
            while len(X) < different_clips_per_batch:
                idxs = list(range(25*5))
                np.random.shuffle(idxs)
                idxs = idxs[:number_of_frames_per_clips] # keep only 2 random frames per clip
                
                print('read clip '+str(samples[i+j])+' at idxs '+str(idxs))
                ok = True

                try:
                    Xj = pims.Video(root_dataset+'/train/X/X'+str(samples[i+j])+'.mp4')[idxs]
                    Xj = np.array(Xj, dtype='float32') / 255.
                
                    Yj = pims.Video(root_dataset+'/train/Y/Y'+str(samples[i+j])+'.mp4')[idxs]
                    Yj = np.array(Yj, dtype='float32') / 255.
                except:
                    print('Error clip number '+ str(samples[i+j]) + ' at '+root_dataset+'/train/X/X'+str(samples[i+j])+'.mp4'+ ' OR '+root_dataset+'/train/Y/Y'+str(samples[i+j])+'.mp4')
                    ok = False
                if ok:
                    X.append(Xj)
                    Y.append(Yj)
                j = j + 1

            # get random non-overlapped patches
            X = np.asarray(X)
            X = X.reshape((X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
            X = X.reshape(-1, fsize//32,32,fsize//32,32, 3).swapaxes(2,3).reshape(-1,32,32,3) 
            
            Y = np.asarray(Y)
            Y = Y.reshape((Y.shape[0]*Y.shape[1], Y.shape[2], Y.shape[3], Y.shape[4]))
            Y = Y.reshape(-1, fsize//32,32,fsize//32,32, 3).swapaxes(2,3).reshape(-1,32,32,3) 
           
            # compute differnce to look for patches including text
            # wrong image comparison, should use opencv or PILLOW, but ok..
            Tt = abs(X - Y)
            T=np.array([np.max(t) for t in Tt])
            T[T>0.2] = 1
            T[T<0.2] = 0
           
            # get random positive and negative patches
            Tpos_idxs = np.where(T>0)[0]
            np.random.shuffle(Tpos_idxs)
            Tneg_idxs = np.where(T==0)[0]
            np.random.shuffle(Tneg_idxs)

            # try to make nbpos = nbneg = batchsize/2
            nbpos = int(batchsize/2)
            if len(Tpos_idxs) < nbpos: nbpos = len(Tpos_idxs)

            # shuffle idxs
            patch_idxs = np.concatenate([Tpos_idxs[:nbpos], Tneg_idxs[:int(batchsize-nbpos)]])
            np.random.shuffle(patch_idxs)
            X = X[patch_idxs]
            Y = Y[patch_idxs]
            T = T[patch_idxs]

            # save in pickle
            data = (X,Y,T)
            with open('batches/batch_'+str(num_batch)+'.pkl', 'wb') as f:
                print('write batch '+str(num_batch))
                pickle.dump(data, f)
                num_batch = num_batch + 1


# load and return minibatches for training
def load_batches(idxfrom, idxto): # 0, 3500
    train_batches = []
    for i in range(idxfrom, idxto):
        with open('batches/batch_'+str(i)+'.pkl', 'rb') as f:
            train_batches.append(pickle.load(f))
    return train_batches

if __name__ == "__main__":
    if sys.argv[1] == 'build_and_save_batches': build_and_save_batches(40000, 128)


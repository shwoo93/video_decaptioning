import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageChops
import os
import math
import functools
import json
import copy

from utils import load_value_file
import random
import cv2
import numpy as np
import pdb

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        image_path2 = os.path.join(video_dir_path, 'image_{:05d}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        elif os.path.exists(image_path2):
            video.append(image_loader(image_path2))
        else:
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

def make_dataset(root_path, subset, n_samples_for_each_video, sample_duration):

    video_input_path = os.path.join(root_path, 'X')
    if subset == 'training':
        video_target_path = os.path.join(root_path, 'Y')
    dataset = []
    for video_name in os.listdir(video_input_path):
        input_path = os.path.join(video_input_path, video_name)
        if subset == 'training':
            target_name = 'Y' + video_name[1:]
            target_path = os.path.join(video_target_path, target_name)
            if not (os.path.exists(input_path) and os.path.exists(target_path)):
                continue
        else:
            target_path = None
        n_frames = 125
        begin_t = 1
        end_t = n_frames
        sample = {
            'video':input_path,
            'segment': [begin_t, end_t],
            'n_frames': 125,
            'video_id':video_name[1:],
            'target_video' :target_path
        }
        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else: # TEST
            if n_samples_for_each_video > 1:
                step = max(1, math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            elif n_samples_for_each_video ==0:
                step = sample_duration

            if n_samples_for_each_video ==-1: # SHRINK 
                for j in range(1, n_frames+1): # 1 ~ 125
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = [j] # j
                    dataset.append(sample_j)
            else:  
                for j in range(1, n_frames, step):
                    sample_j = copy.deepcopy(sample)
                    sample_j['frame_indices'] = list( range(j, min(n_frames + 1, j + sample_duration)))
                    dataset.append(sample_j)

    return dataset

class VideoDecaptionData(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 opt=None):
        self.subset = subset
        if subset == 'testing':
            n_samples_for_each_video = -1
        self.data = make_dataset(root_path, subset, n_samples_for_each_video, sample_duration)

        self.spatial_transform = spatial_transform
        self.target_transform = spatial_transform
        self.temporal_transform = temporal_transform

        self.loader = get_loader()

        self.opt = opt
        self.lr_flip = opt.lr_flip
        self.tb_flip = opt.tb_flip
        self.t_stride = opt.t_stride
        self.t_shrink = opt.t_shrink
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.subset == 'training':
            path = self.data[index]['video']
            target_path = self.data[index]['target_video']
            frame_indices = self.data[index]['frame_indices']

            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
                if self.t_stride > 1:
                    frame_indices = frame_indices[0::self.t_stride]

            clip = self.loader(path, frame_indices)
            if self.t_shrink:
                mid_idx = len(frame_indices)//2
                target_indices = [frame_indices[mid_idx]]
                target_clip = self.loader(target_path, target_indices)
            else:
                target_clip = self.loader(target_path, frame_indices)

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
                target_clip = [self.spatial_transform(img) for img in target_clip]

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            target_clip = torch.stack(target_clip, 0).permute(1, 0, 2, 3)

            """ Random Horizontal & Vertical Flip """
            if self.lr_flip and random.random() < 0.5:
                idx = [i for i in range(clip.size(3)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                clip = clip.index_select(3, idx)
                target_clip = target_clip.index_select(3, idx)
            if self.tb_flip and random.random() < 0.5:
                idx = [i for i in range(clip.size(2)-1, -1, -1)]
                idx = torch.LongTensor(idx)
                clip = clip.index_select(2, idx)
                target_clip = target_clip.index_select(2, idx)

            return clip, target_clip

        else:
            path = self.data[index]['video']
            frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                if self.t_shrink:
                    frame_indices_ = []
                    for i in range(self.opt.sample_duration):
                        frame_indices_.append(frame_indices[0]+self.t_stride*i)
                    offset = (len(frame_indices_)//2)*self.t_stride
                    frame_indices = [x-offset for x in frame_indices_]
                    frame_indices = [-x+2 if x<=0 else x for x in frame_indices]
                    frame_indices = [125*2-x if x>125 else x for x in frame_indices]
                else:
                    frame_indices = self.temporal_transform(frame_indices)
                    if self.t_stride > 1:
                        frame_indices = frame_indices[0::self.t_stride]

            clip = self.loader(path, frame_indices)

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
                
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            return clip, path
        
    def __len__(self):
        return len(self.data)
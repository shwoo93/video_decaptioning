import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
from matplotlib import colors

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, value):
        for t in self.transforms:
            img = t(img, value)
        return img

class RandomContrast(object):
    def __init__(self, lower=0.3, upper=0.9):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, value):
        if random.randint(2):
            # alpha = random.uniform(self.lower, self.upper)
            # image *= alpha
            image *= value[0]
        return image

class ConvertColor(object):
    def __init__(self, current='RGB', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image,  value):
        if self.current == 'RGB' and self.transform == 'HSV':
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = colors.rgb_to_hsv(image)
        elif self.current == 'HSV' and self.transform == 'RGB':
            # image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            image = colors.hsv_to_rgb(image)
        else:
            raise NotImplementedError
        return image

class RandomSaturation(object):
    def __init__(self, lower=0.3, upper=0.9):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, value):
        # if random.randint(2):
            # image[:, :, 1] *= random.uniform(self.lower, self.upper)
        image[:, :, 1] *= value[1]
        return image

class PhotometricDistort(object):
    def __init__(self):
        # self.pd = [
        #     ConvertColor(transform='HSV'),
        #     RandomSaturation(),
        #     ConvertColor(current='HSV', transform='RGB'),
        #     RandomContrast(),
        # ]
        self.random_contrast = RandomContrast()
        self.contrast_value = random.uniform(0.5, 0.55)
        self.saturation_value = random.uniform(0.3, 0.35)
    def __call__(self, images):
        # distort = Compose(self.pd)
        if random.randint(2):
            value = [self.contrast_value, self.saturation_value]
            distorted_images = []
            for im in images:
                # im = distort(im, value)
                im = self.random_contrast(im, value)
                distorted_images.append(im)
        else:
            distorted_images = images
        # return im
        return distorted_images

    def randomize_parameters(self):
        pass
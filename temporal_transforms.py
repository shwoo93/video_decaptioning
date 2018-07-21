import random
import math
import pdb

class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices
        print(len(out))
        for index in out:
            print(index)
            if len(out) >= self.size:
                break
            out.append(index)
        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCropMirror(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        mid = self.size//2
        tmp1 = frame_indices[1:mid][::-1] # 31
        tmp2 = frame_indices[-mid:-1][::-1] # 31
        frame_indices = tmp1+frame_indices+tmp2
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

class TemporalRandomCut(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        if len(frame_indices) < self.size:
            out = frame_indices
            for index in out:
                if len(out) >= (self.size/2): #len(out) >= self.size/2(=64)
                    break
                out.append(index)
        else:
            rand_end = max(0, len(frame_indices) - self.size - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.size, len(frame_indices))

            out = frame_indices[begin_index:end_index]

            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)

        return out


class RandomTemporalFlip(object):
    """temporally flip given volume randomly with a probability of 0.5."""

    def __call__(self, frame_indices):
        """
        Args:
            video clip to be flipped.
        Returns:
            randomly flipped video clip.
        """
        if random.random() < 0.5:
            return frame_indices[::-1]
        return frame_indices

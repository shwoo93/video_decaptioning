import h5py
import numpy as np
import torch
from torch.autograd import Variable
import os
import dill as pkl
from itertools import tee
from torch import nn
import pdb

def clip_grad_norm(named_parameters, max_norm, clip=False, verbose=False):
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        # for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
        #     print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
        # print('-------------------------------', flush=True)

    return total_norm
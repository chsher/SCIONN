import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_gumbel(shape, device, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, dtype=torch.float32, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, device): 
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, device, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probability distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, device)
    if hard:
        y = F.one_hot(torch.argmax(y, dim=-1), y.shape[-1])
    return y

def update_tile_shape(H_in, kernel_size, W_in=None, dilation=1.0, padding=0.0, stride=1.0):
    H_out = (H_in + 2.0 * padding - dilation * (kernel_size - 1) - 1) / stride + 1

    if W_in is not None:
        W_out = (W_in + 2.0 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        return int(np.floor(H_out)), int(np.floor(W_out))
    else:
        return int(np.floor(H_out))
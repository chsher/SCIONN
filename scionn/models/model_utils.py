import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.models import logreg, rnnet, scionnet

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_model(net_name, net_params, input_size, seq_len, device, outfilek, statsfile=None, kidx=None, attr=False):
    if net_name == 'logreg':
        output_size, dropout = net_params
        net = logreg.LogReg(input_size, seq_len, output_size, dropout)
        lamb, temp, gumbel, adv = np.nan, np.nan, False, False

    elif net_name in ['rnnet', 'gru', 'lstm']:
        output_size, hidden_size, n_layers, dropout, bidirectional, agg, hide = net_params
        net = rnnet.RNNet(input_size, hidden_size, output_size, n_layers, dropout, bidirectional, agg, hide, net_name=net_name)
        lamb, temp, gumbel, adv = np.nan, np.nan, False, False

    elif net_name == 'scionnet':
        output_size, n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, lamb, temp, adv, hard, dropout = net_params

        if os.path.exists(statsfile):
            with open(statsfile, 'rb') as f:
                while 1:
                    try:
                        stats = pickle.load(f)
                        if stats['k'] == kidx and stats['split'] in ['val', 'test']:
                            lamb = stats['lamb_best']
                            temp = stats['temp_best']
                            break
                    except EOFError:
                        break

        net = scionnet.SCIONNet(n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, temp, device, adv=adv, hard=hard, 
            dropout=dropout, in_channels=input_size, out_channels=output_size, H_in=seq_len, hide=attr)
    
    if os.path.exists(outfilek):
        saved_state = torch.load(outfilek, map_location=lambda storage, loc: storage)
        net.load_state_dict(saved_state)

    return net, lamb, temp, gumbel, adv

def load_sk_model(net_name, outfilek, random_state):
    if net_name == 'LR':
            lr = LogisticRegression(random_state=random_state, n_jobs=-1)
    elif net_name == 'RF':
        lr = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    if os.path.exists(outfilek):
        with open(outfilek, 'rb') as f:
            lr = pickle.load(f)

    return lr

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
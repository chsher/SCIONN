import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class SingleCellDataset(Dataset):
    def __init__(self, adata, pt_idxs, pt_labels, idxs, batch_size, seq_len, input_size, trainBaseline=False, baselineStats=None, multiplier=2, 
        random_state=None, details=False, returnBase=False):

        self.adata = adata
        self.pt_idxs = pt_idxs
        self.pt_labels = pt_labels
        self.idxs = idxs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_size = input_size
        self.trainBaseline = trainBaseline
        self.baselineStats = baselineStats
        self.multiplier = multiplier
        self.random_state = random_state
        self.details = details
        self.returnBase = returnBase

        try:
            self.n_labels = len(set(self.pt_labels.values()))
        except:
            self.n_labels = len(list(self.pt_labels.values())[0])

        if type(self.random_state) is int:
            np.random.seed(self.random_state)

        if type(self.seq_len) == dict:
            self.ms = [k for k, v in self.seq_len.items() if v > 0]
        else:
            self.ms = None

        if self.trainBaseline:
            if 'stdev' in self.baselineStats.keys() and 'mean' in self.baselineStats.keys():
                self.xb = torch.FloatTensor(np.nan_to_num((0.0 - self.baselineStats['mean']) / self.baselineStats['stdev']))
            elif 'mean' in self.baselineStats.keys():
                self.xb = torch.FloatTensor(self.baselineStats['mean'])
            else:
                empty_input = torch.empty(self.input_size)
                self.xb = torch.zeros_like(empty_input) 

            self.xb = self.xb.repeat(self.seq_len, 1)

    def __len__(self):    
        if self.trainBaseline:
            return self.batch_size * 2
        else:
            return self.batch_size

    def __getitem__(self, idx):
        if self.trainBaseline and idx >= len(self.idxs):
            x = self.xb
            y = torch.FloatTensor([idx % self.n_labels])
            b_idxs = None
            base = torch.FloatTensor([1.0])

        else:
            idx = int(idx % len(self.idxs))
            b = self.idxs[idx]

            if self.ms is not None:
                b_idxs = np.concatenate([np.random.choice(self.pt_idxs[b][m], self.seq_len[m], replace=True) for m in ms])
            else:
                replace = self.seq_len > len(self.pt_idxs[b])
                b_idxs = np.random.choice(self.pt_idxs[b], self.seq_len, replace=replace)

            x = torch.FloatTensor(self.adata[b_idxs, :].X.todense().reshape(-1, self.input_size))
            y = torch.FloatTensor([self.pt_labels[b]])
            base = torch.FloatTensor([0.0])
            
        if self.details:
            return x, y, b_idxs
        elif self.returnBase:
            return x, y, base.repeat(seq_len)
        else:
            return x, y
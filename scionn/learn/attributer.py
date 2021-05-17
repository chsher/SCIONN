import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pickle
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm, trange
from captum.attr import IntegratedGradients

import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import data_utils
from scionn.models import model_utils
from scionn.learn import trainer

import pdb

MMR_TO_IDX = {
    'MSS': 0,
    'MSI': 1
}

def check_baseline_training(adata, label, seq_len, batch_size, net_name, net_params, outfile, statsfile, device, kfold=10, ylabel='MMRLabel', 
    ptlabel='PatientBarcode', smlabel='PatientTypeID', scale=True, returnBase=True, bdata=None, random_state=32921, verbose=True):

    input_size = adata.shape[1]
    loss_fn = nn.BCEWithLogitsLoss()

    splits_msi, splits_mss, idxs_msi, idxs_mss = data_utils.make_splits(adata, ylabel, ptlabel, kfold, random_state=random_state)

    for kidx in trange(kfold):
        a = outfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        outfilek = '.'.join(a)

        datasets = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, kfold, ylabel, ptlabel, smlabel, 
            scale=scale, returnBase=returnBase, baseOnly=True, bdata=bdata, random_state=random_state)
        loaders = [DataLoader(d, batch_size=len(d), shuffle=False, pin_memory=True, drop_last=False) for d in datasets]
        
        net, lamb, temp, gumbel, adv = model_utils.load_model(net_name, net_params, input_size, seq_len, device, outfilek, statsfile=statsfile, kidx=kidx)
        net.temp = 0.1

        for loader, loader_label in zip(loaders, ['train_base', 'val_base', 'test_base']):
            loss, auc, frac_tiles = trainer.run_validation_loop(0, loader, net, loss_fn, device, lamb=lamb, temp=0.1, gumbel=gumbel, adv=adv, verbose=verbose, blabel=loader_label)
            
            with open(statsfile, 'ab') as f:
                pickle.dump({'k': kidx, 
                    'split': loader_label,
                    'loss_best': loss, 
                    'auc_best': auc, 
                    'frac_best': frac_tiles, 
                    'lamb_best': lamb, 
                    'temp_best': temp}, f)

def run_integrated_gradients(adata, label, seq_len, net_name, net_params, outfile, statsfile, attrfile, device, kfold=10, ylabel='MMRLabel', 
    ptlabel='PatientBarcode', smlabel='PatientTypeID', ctlabel='v11_bot', scale=True, trainBaseline=True, returnBase=True, bdata=None, random_state=32921, verbose=True):

    input_size = adata.shape[1]
    adata.obs[ylabel] = adata.obs[label].apply(lambda x: MMR_TO_IDX[x])

    splits_msi, splits_mss, idxs_msi, idxs_mss = data_utils.make_splits(adata, ylabel, ptlabel, kfold, random_state=random_state)

    for kidx in trange(kfold):
        a = outfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        outfilek = '.'.join(a)

        datasets = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, kfold, ylabel, ptlabel, smlabel, 
            scale=scale, trainBaseline=False, returnBase=False, details=True, bdata=bdata, random_state=random_state)
        xb = train.xb

        net, lamb, temp, gumbel, adv = model_utils.load_model(net_name, net_params, input_size, seq_len, device, outfilek, statsfile=statsfile, kidx=kidx, attr=True)
        net.temp = 0.1

        net.train()
        net.to(device)

        coeffs = pd.DataFrame()

        for dataset in datasets:
            for i in trange(len(dataset)):
                x, y, b = dataset.__getitem__(i)
                input = x.view(1, x.shape[0], x.shape[1])
                baseline = xb.view(1, x.shape[0], x.shape[1])

                igd = IntegratedGradients(net)
                attributions, delta = igd.attribute(input.to(device), baseline.to(device), return_convergence_delta=True)

                new_ids = np.array(b).flatten()
                new_ats = attributions.cpu().numpy().reshape((-1, adata.shape[1]))
                
                df = pd.DataFrame(np.concatenate((new_ids[..., np.newaxis], new_ats), axis=1), columns=['idx'] + list(adata.var.index))
                df['cl'] = df['idx'].apply(lambda x: dataset.adata.obs.loc[dataset.adata.obs.index[int(x)], ctlabel])
                df['y'] = df['idx'].apply(lambda x: dataset.adata.obs.loc[dataset.adata.obs.index[int(x)], ylabel])
                df['pt'] = df['idx'].apply(lambda x: dataset.adata.obs.loc[dataset.adata.obs.index[int(x)], smlabel])
                
                coeffs2 = df.iloc[:, 1:].groupby(['cl', 'y', 'pt']).mean().fillna(0)
                coeffs3 = coeffs2.swaplevel(i=0, j=1).unstack(1).fillna(0)
                coeffs = pd.concat([coeffs, coeffs3])

        a = attrfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        attrfilek = '.'.join(a)

        coeffs.fillna(0, inplace=True)
        coeffs.to_csv(attrfilek)
import torch
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm, trange
from captum.attr import IntegratedGradients

import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import data_utils
from scionn.models import logreg, rnnet, scionnet

import pdb

MMR_TO_IDX = {
    'MSS': 0,
    'MSI': 1
}

def run_integrated_gradients(adata, label, seq_len, net_name, net_params, outfile, attrfile, device, kfold=10, ylabel='MMRLabel', ptlabel='PatientBarcode', 
    smlabel='PatientTypeID', ctlabel='v11_bot', scale=True, trainBaseline=True, returnBase=True, random_state=32921, verbose=True):

    input_size = adata.shape[1]
    adata.obs[ylabel] = adata.obs[label].apply(lambda x: MMR_TO_IDX[x])

    splits_msi, splits_mss, idxs_msi, idxs_mss = data_utils.make_splits(adata, ylabel, ptlabel, kfold, random_state=random_state)

    for kidx in trange(kfold):
        a = outfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        outfilek = '.'.join(a)

        train, val = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, ylabel, ptlabel, smlabel, 
            scale=scale, trainBaseline=False, returnBase=False, details=True, train_only=False, random_state=random_state)
        xb = train.xb

        if net_name == 'logreg':
            output_size, dropout = net_params
            net = logreg.LogReg(input_size, seq_len, output_size, dropout)
            lamb, temp, gumbel, adv = None, None, False, False
        elif net_name in ['rnnet', 'gru', 'lstm']:
            output_size, hidden_size, n_layers, dropout, bidirectional, agg, hide = net_params
            net = rnnet.RNNet(input_size, hidden_size, output_size, n_layers, dropout, bidirectional, agg, hide, net_name=net_name)
            lamb, temp, gumbel, adv = None, None, False, False
        elif net_name == 'scionnet':
            output_size, n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, lamb, temp, adv, hard, dropout = net_params
            net = scionnet.SCIONNet(n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, temp, device, adv=adv, hard=hard, dropout=dropout, in_channels=input_size, out_channels=output_size, H_in=seq_len)
        
        net.eval()
        net.to(device)

        if os.path.exists(outfilek):
            saved_state = torch.load(outfilek, map_location=lambda storage, loc: storage)
            net.load_state_dict(saved_state)

        coeffs = pd.DataFrame()

        for dataset in [train, val]:
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
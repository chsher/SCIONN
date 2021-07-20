import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import pickle
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm, trange
from captum.attr import IntegratedGradients

import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import data_utils
from scionn.models import model_utils
from scionn.learn import learn_utils
from scionn.learn import trainer

import pdb

def check_baseline_training(adata, label, seq_len, batch_size, net_name, net_params, outfile, statsfile, device, kfold=10, ylabel='MMRLabel', 
    ptlabel='PatientBarcode', smlabel='PatientTypeID', scale=True, returnBase=True, bdata=None, pin_memory=True, n_workers=0, random_state=32921, 
    verbose=True, catlabel=None, split_by_study=False):

    input_size = adata.shape[1]

    if ylabel in ['MMRLabel', 'ResponseLabel']:
        loss_fn = nn.BCEWithLogitsLoss()
        metric = 'auc'
    elif ylabel in ['InFreq', 'InFreqPD1']:
        loss_fn = learn_utils.KLDivLoss()
        metric = 'mae'

    num_embeddings = max(adata.obs[catlabel]) if catlabel is not None else None

    splits_msi, splits_mss, idxs_msi, idxs_mss, kfold = data_utils.make_splits(adata, ylabel, ptlabel, kfold, split_by_study=split_by_study, random_state=random_state)

    for kidx in trange(kfold):
        a = outfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        outfilek = '.'.join(a)

        datasets = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, kfold, ylabel, ptlabel, smlabel, 
            batch_size=batch_size, scale=scale, returnBase=returnBase, baseOnly=True, bdata=bdata, random_state=random_state, catlabel=catlabel)
        loaders = [DataLoader(d, batch_size=len(d), shuffle=False, pin_memory=pin_memory, num_workers=n_workers, drop_last=False) for d in datasets]
        
        net, lamb, temp, gumbel, adv = model_utils.load_model(net_name, net_params, input_size, seq_len, device, outfilek, statsfile=statsfile, kidx=kidx,
            num_embeddings=num_embeddings)
        if net_name == 'scionnet':
            net.temp = 0.1

        for loader, loader_label in zip(loaders, ['train_base', 'val_base', 'test_base']):
            loss, auc, frac_tiles = trainer.run_validation_loop(0, loader, net, loss_fn, device, lamb=lamb, temp=0.1, gumbel=gumbel, adv=adv, verbose=verbose, blabel=loader_label)
            
            with open(statsfile, 'ab') as f:
                pickle.dump({'k': kidx, 
                    'split': loader_label,
                    'loss_best': loss, 
                    metric + '_best': auc, 
                    'frac_best': frac_tiles, 
                    'lamb_best': lamb, 
                    'temp_best': temp}, f)

def run_integrated_gradients(adata, label, seq_len, net_name, net_params, outfile, statsfile, attrfile, device, kfold=10, ylabel='MMRLabel', 
    ptlabel='PatientBarcode', smlabel='PatientTypeID', ctlabel='v11_bot', scale=True, trainBaseline=True, returnBase=True, bdata=None, 
    num_replicates=10, random_state=32921, verbose=True, catlabel=None, split_by_study=False, freqlabel='v11_bot'):

    net_params[-1] = 0.0
    input_size = adata.shape[1]

    mapper = {'InFreq': {'v11_mid': 13, 'v11_top': 5}, 'InFreqPD1': 19}
    num_targets = mapper.get(ylabel, 1)

    if type(num_targets) is dict:
        num_targets = num_targets.get(freqlabel, 1)

    num_embeddings = max(adata.obs[catlabel]) if catlabel is not None else None

    splits_msi, splits_mss, idxs_msi, idxs_mss, kfold = data_utils.make_splits(adata, ylabel, ptlabel, kfold, split_by_study=split_by_study, random_state=random_state)

    for kidx in trange(kfold):
        a = attrfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        attrfilek = '.'.join(a)

        if not os.path.exists(attrfilek):
            a = outfile.split('.')
            a[0] = a[0] + '_' + str(kidx).zfill(2)
            outfilek = '.'.join(a)

            datasets = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, kfold, ylabel, ptlabel, smlabel, 
                scale=scale, trainBaseline=False, returnBase=False, details=True, bdata=bdata, random_state=random_state, catlabel=catlabel)
            xb = datasets[0].xb

            net, lamb, temp, gumbel, adv = model_utils.load_model(net_name, net_params, input_size, seq_len, device, outfilek, statsfile=statsfile, kidx=kidx, 
                num_embeddings=num_embeddings, attr=True)
            if net_name == 'scionnet':
                net.temp = 0.1

            net.train()
            net.to(device)

            coeffs = pd.DataFrame()

            for nr in trange(num_replicates):
                for dataset in datasets:
                    for i in trange(len(dataset)):
                        x, y, b = dataset.__getitem__(i)
                        input = x.view(1, x.shape[0], x.shape[1])

                        #if catlabel is None:
                        #    baseline = xb.view(1, x.shape[0], x.shape[1])
                        #else:
                        #    x_cat = torch.FloatTensor(np.random.choice(np.arange(num_embeddings), seq_len, replace=True))
                        #    baseline = torch.cat((xb.view(1, x.shape[0], x.shape[1] - 1), x_cat.unsqueeze(0).unsqueeze(-1)), dim=-1)
                        baseline = xb.view(1, x.shape[0], x.shape[1])

                        for target in range(num_targets):
                            igd = IntegratedGradients(net)
                            if ylabel in ['MMRLabel', 'ResponseLabel']:
                                attributions, delta = igd.attribute(input.to(device), baseline.to(device), return_convergence_delta=True)
                            elif ylabel in ['InFreq', 'InFreqPD1']:
                                attributions, delta = igd.attribute(input.to(device), baseline.to(device), target=target, return_convergence_delta=True)

                            new_ids = np.array(b).flatten()
                            new_ats = attributions.cpu().numpy().reshape((-1, adata.shape[1])) #+ (1 if catlabel is not None else 0)
                            
                            df = pd.DataFrame(np.concatenate((new_ids[..., np.newaxis], new_ats), axis=1), columns=['idx'] + list(adata.var.index)) #+ ([catlabel] if catlabel is not None else [])
                            df['cl'] = df['idx'].apply(lambda x: dataset.adata.obs.loc[dataset.adata.obs.index[int(x)], ctlabel])
                            df['pt'] = df['idx'].apply(lambda x: dataset.adata.obs.loc[dataset.adata.obs.index[int(x)], smlabel])
                            
                            if ylabel in ['MMRLabel', 'ResponseLabel']:
                                df['y'] = df['idx'].apply(lambda x: dataset.adata.obs.loc[dataset.adata.obs.index[int(x)], ylabel])
                                coeffs2 = df.iloc[:, 1:].groupby(['cl', 'y', 'pt']).mean().fillna(0)
                            elif ylabel in ['InFreq', 'InFreqPD1']:
                                df['tgt'] = target
                                coeffs2 = df.iloc[:, 1:].groupby(['cl', 'tgt', 'pt']).mean().fillna(0)
                            
                            coeffs3 = coeffs2.swaplevel(i=0, j=1).unstack(1).fillna(0)
                            coeffs = pd.concat([coeffs, coeffs3])

            coeffs.fillna(0, inplace=True)
            coeffs.to_csv(attrfilek)

        else:
            print('Attribution scores already found, k={}'.format(kidx))

def compare_groundtruth(kfold, groundtruth1, groundtruth2, statsfile, attrfile, compare_sd=False, ctlabel='group', addlabel=False):
    
    if ctlabel == 'group':
        cl = ['1', '2']
    elif ctlabel == 'leiden':
        cl = ['8', '7']

    if addlabel:
        groundtruth1 = list(groundtruth1) + [ctlabel + cl[0]]
        groundtruth2 = list(groundtruth2) + [ctlabel + cl[1]]

    n = len(groundtruth1) + len(groundtruth2)

    for kidx in trange(kfold):
        a = attrfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        attrfilek = '.'.join(a)

        try:
            coeffs = pd.read_csv(attrfilek, index_col=[0,1], header=[0,1])
        except:
            print('Attribution file not found, k={}'.format(kidx))
            continue

        coeffs = coeffs.loc[(1, ), ]

        t = coeffs.sum(axis=0) 
        keep = t.sort_values(ascending=False)[:n].index
        total = np.sum([(i[0] in groundtruth1 and i[1] == cl[0]) or (i[0] in groundtruth2 and i[1] == cl[1]) for i in keep])

        print('k: {0:2}, top {1:3}, n_correct: {2:3}, n_total: {3:3}, fr_correct: {4:.4f}'.format(kidx, n, total, n, total / n))
        with open(statsfile, 'ab') as f:
            pickle.dump([kidx, n, total, n, total / n], f)

        for gt, c in zip([groundtruth1, groundtruth2], cl):
            total = np.sum([i[0] in gt and i[1] == c for i in keep]) 
            print('k: {0:2}, top {1:3}, n_correct: {2:3}, n_total: {3:3}, fr_correct: {4:.4f}'.format(kidx, n // 2, total, n // 2, total / (n // 2)))
            with open(statsfile, 'ab') as f:
                pickle.dump([kidx, n // 2, total, n // 2, total / (n // 2)], f)

        if compare_sd:
            m = t.mean()
            s = t.std()

            for sd in [0.5, 1.0, 1.5, 2.0]:
                keep = coeffs.columns[(t > m + sd * s)]

                if len(keep) > 0:
                    total = np.sum([(i[0] in groundtruth1 and i[1] == cl[0]) or (i[0] in groundtruth2 and i[1] == cl[1]) for i in keep])
                    print('k: {0:2}, sd: {1:.1f}, n_correct: {2:3}, n_total: {3:3}, fr_correct: {4:.4f}'.format(kidx, sd, total, len(keep), total / len(keep)))

                    with open(statsfile, 'ab') as f:
                        pickle.dump([kidx, sd, total, len(keep), total / len(keep)], f)
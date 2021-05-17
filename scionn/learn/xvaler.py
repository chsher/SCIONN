import scipy
import pickle
import numpy as np
import anndata as ad
from tqdm.autonotebook import tqdm, trange
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import data_utils
from scionn.models import model_utils
from scionn.learn import trainer

import torch
import torch.nn as nn
import torch.optim as optim

import pdb

def run_kfold_xvalidation(adata, label, seq_len, batch_size, net_name, net_params, learning_rate, weight_decay, patience, num_epochs, outfile, statsfile, device, kfold=10, 
    ylabel='MMRLabel', ptlabel='PatientBarcode', smlabel='PatientTypeID', training=True, validate=True, scale=True, trainBaseline=True, returnBase=True, bdata=None, random_state=32921, verbose=True):

    input_size = adata.shape[1]
    loss_fn = nn.BCEWithLogitsLoss()

    splits_msi, splits_mss, idxs_msi, idxs_mss = data_utils.make_splits(adata, ylabel, ptlabel, kfold, bdata=bdata, random_state=random_state)

    for kidx in trange(kfold):
        a = outfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        outfilek = '.'.join(a)

        datasets = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, kfold, ylabel, ptlabel, smlabel, 
            scale=scale, trainBaseline=trainBaseline, returnBase=returnBase, bdata=bdata, random_state=random_state)
        
        if len(datasets) == 3:
            train, val, test = datasets
            train_loader = DataLoader(train, batch_size=len(train), shuffle=True, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val, batch_size=len(val), shuffle=False, pin_memory=True, drop_last=False)
            test_loader = DataLoader(test, batch_size=len(test), shuffle=False, pin_memory=True, drop_last=False)
        elif len(datasets) == 2:
            train, test = datasets
            train_loader = DataLoader(train, batch_size=len(train), shuffle=True, pin_memory=True, drop_last=True)            
            test_loader = DataLoader(test, batch_size=len(test), shuffle=False, pin_memory=True, drop_last=False)

        net, lamb, temp, gumbel, adv = model_utils.load_model(net_name, net_params, input_size, seq_len, device, outfilek, statsfile=statsfile, kidx=kidx)

        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=verbose)

        lamb_orig, temp_orig, best_loss, best_epoch, best_auc, best_fr, best_lamb, best_temp, tally = lamb, temp, 1e09, 0, 0, 0, lamb, temp, 0
        
        if training or validate:
            for e in tqdm(range(num_epochs)):
                if training:
                    trainer.run_training_loop(e, train_loader, net, loss_fn, optimizer, device, lamb=lamb, temp=temp, gumbel=gumbel, adv=adv, verbose=verbose)
                
                loss, auc, frac_tiles = trainer.run_validation_loop(e, val_loader, net, loss_fn, device, lamb=lamb, temp=temp, gumbel=gumbel, adv=adv, verbose=verbose)
                
                scheduler.step(loss)

                if loss < best_loss: 
                    tally = 0
                    best_loss, best_epoch, best_auc, best_fr, best_lamb, best_temp = loss, e, auc, frac_tiles, lamb, temp
                    print('New best val epoch: {0:d}, loss: {1:.4f}, AUC: {2:.4f}, frac: {3:.4f}, lamb: {4:.4f}, temp: {5:.4f}'.format(best_epoch, best_loss, best_auc, best_fr, best_lamb, best_temp))
                    torch.save(net.state_dict(), outfilek)
                else:
                    tally += 1

                if tally > patience:
                    tally = 0
                    saved_state = torch.load(outfilek, map_location=lambda storage, loc: storage)
                    net.load_state_dict(saved_state)
                    print('Reloaded model epoch:', best_epoch)

                if (lamb is not None and temp is not None) and (e > 50 and e % patience == 0):
                    lamb += 0.01
                    lamb = min(lamb, 1.0)
                    print('Increased lambda:', lamb)

                    temp -= 0.1
                    temp = max(temp, 0.1)
                    print('Reduced temperature:', temp)
            
            print('Overall best val epoch: {0:d}, loss: {1:.4f}, AUC: {2:.4f}, frac: {3:.4f}, lamb: {4:.4f}, temp: {5:.4f}'.format(best_epoch, best_loss, best_auc, best_fr, best_lamb, best_temp))

            with open(statsfile, 'ab') as f:
                pickle.dump({'k': kidx, 
                    'split': 'val',
                    'lamb_start': lamb_orig, 
                    'lamb_end': lamb, 
                    'temp_start': temp_orig, 
                    'temp_end': temp, 
                    'lr_start': learning_rate, 
                    'lr_end': optimizer.param_groups[0]['lr'], 
                    'epoch_best': best_epoch, 
                    'loss_best': best_loss, 
                    'auc_best': best_auc, 
                    'frac_best': best_fr, 
                    'lamb_best': best_lamb, 
                    'temp_best': best_temp}, f)

        blabel = 'Test' if bdata is None else 'Ext'
        saved_state = torch.load(outfilek, map_location=lambda storage, loc: storage)
        net.load_state_dict(saved_state)
        loss, auc, frac_tiles = trainer.run_validation_loop(e, test_loader, net, loss_fn, device, lamb=best_lamb, temp=best_temp, gumbel=gumbel, 
            adv=adv, verbose=verbose, blabel=blabel)

        print('Overall best test epoch: {0:d}, loss: {1:.4f}, AUC: {2:.4f}, frac: {3:.4f}, lamb: {4:.4f}, temp: {5:.4f}'.format(best_epoch, loss, auc, frac_tiles, best_lamb, best_temp))

        with open(statsfile, 'ab') as f:
            pickle.dump({'k': kidx, 
                'split': 'test',
                'lamb_start': lamb_orig, 
                'lamb_end': lamb, 
                'temp_start': temp_orig, 
                'temp_end': temp, 
                'lr_start': learning_rate, 
                'lr_end': optimizer.param_groups[0]['lr'], 
                'epoch_best': best_epoch, 
                'loss_best': loss, 
                'auc_best': auc, 
                'frac_best': frac_tiles, 
                'lamb_best': best_lamb, 
                'temp_best': best_temp}, f)
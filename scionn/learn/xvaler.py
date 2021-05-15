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
from scionn.models import logreg, rnnet, scionnet
from scionn.learn import trainer

import torch
import torch.nn as nn
import torch.optim as optim

def run_kfold_xvalidation(adata, label, seq_len, batch_size, net_name, net_params, learning_rate, weight_decay, patience, num_epochs, outfile, statsfile, device, kfold=10, 
    ylabel='MMRLabel', ptlabel='PatientBarcode', smlabel='PatientTypeID', training=True, scale=True, trainBaseline=True, returnBase=True, random_state=32921, verbose=True):

    input_size = adata.shape[1]
    loss_fn = nn.BCEWithLogitsLoss()

    splits_msi, splits_mss, idxs_msi, idxs_mss = data_utils.make_splits(adata, ylabel, ptlabel, kfold, random_state=random_state)

    for kidx in trange(kfold):
        a = outfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        outfilek = '.'.join(a)

        train, val = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, ylabel, ptlabel, smlabel, 
            scale=scale, trainBaseline=trainBaseline, returnBase=returnBase, train_only=False, random_state=random_state)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

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
            net = scionnet.SCIONNet(n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, temp, device, 
                adv=adv, hard=hard, dropout=dropout, in_channels=input_size, out_channels=output_size, H_in=seq_len)
        
        if os.path.exists(outfilek):
            saved_state = torch.load(outfilek, map_location=lambda storage, loc: storage)
            net.load_state_dict(saved_state)

        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=verbose)

        lamb_orig, temp_orig, best_loss, best_epoch, best_auc, best_fr, tally = lamb, temp, 1e09, 0, 0, 0, 0
        
        for e in tqdm(range(num_epochs)):
            if training:
                trainer.run_training_loop(e, train_loader, net, loss_fn, optimizer, device, lamb=lamb, temp=temp, gumbel=gumbel, adv=adv, verbose=verbose)
            
            loss, auc, frac_tiles = trainer.run_validation_loop(e, val_loader, net, loss_fn, device, lamb=lamb, temp=temp, gumbel=gumbel, adv=adv, verbose=verbose)
            
            scheduler.step(loss)

            if loss < best_loss: 
                tally = 0
                best_loss, best_epoch, best_auc, best_fr = loss, e, auc, frac_tiles
                print('New best loss: {0:.4f}, New best AUC: {1:.4f}, New best frac: {2:.4f}'.format(best_loss, best_auc, best_fr))
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
        
        print('Best epoch:', best_epoch, ', Best loss:', best_loss, ', Best AUC:', best_auc, ', Best frac:', best_fr)

        with open(statsfile, 'ab') as f:
            pickle.dump([lamb_orig, lamb, temp_orig, temp, learning_rate, optimizer.param_groups[0]['lr'], best_epoch, best_loss, best_auc, best_fr], f)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

import pickle
import numpy as np

import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.learn import trainer

PRINT_STMT = '{6} best {7} epoch: {0:d}, loss: {1:.4f}, {8}: {2:.4f}, frac: {3:.4f}, lamb: {4:.4f}, temp: {5:.4f}'

class KLDivLoss(_Loss):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_pred, y_true):
        y_pred = F.log_softmax(y_pred, dim=-1)
        kldiv = self.loss(y_pred, y_true)
        return kldiv

def calc_per_smp_stats(smps, kidx, net, device, gumbel, statsfile, outfilek, pin_memory, n_workers, metric='AUC'):
    saved_state = torch.load(outfilek, map_location=lambda storage, loc: storage)
    net.load_state_dict(saved_state)

    with open(statsfile, 'rb') as f:
        while 1:
            try:
                stats = pickle.load(f)
                if stats['k'] == kidx and stats['split'] == 'val':
                    best_lamb, best_temp = stats['lamb_best'], stats['temp_best']
            except EOFError:
                break

    print('{0:>6} | {1:>12} | {2:>6} | {3:>6} | {4:>6} | {5:>6}'.format('Split', 'Sample', 'Label', 'Count', 'Frac', 'Score'))
    
    stats = {}
    net.eval()
    net.to(device)

    for split, smp_list in smps.items():
        stats[split] = {}

        for smp in smp_list:
            loader = DataLoader(smp, batch_size=len(smp), shuffle=False, pin_memory=pin_memory, num_workers=n_workers, drop_last=False)

            for b, (input, label, base) in enumerate(loader):
                input, label, base = input.to(device), label.to(device), base.to(device)
                outputs = net(input)

                if gumbel:
                    output_keep, keep = outputs
                    znorm = torch.sum(torch.max(keep[:, :, -1], base))
                    frac_tiles = znorm / (input.shape[0] * input.shape[1])
                else:
                    output_keep = outputs
                    frac_tiles = 1.0
                
                if label.shape[1] == 1:
                    y_prob = torch.sigmoid(output_keep.detach())
                    score = np.mean(y_prob.cpu().squeeze(-1).numpy())
                    y = label[0]
                else:
                    y_prob = F.softmax(output_keep.detach(), dim=-1)
                    score = np.mean(np.abs(label.cpu().squeeze(-1).numpy() - y_prob.cpu().squeeze(-1).numpy()))
                    y = np.nan

                stats[split][smp.idxs[0]] = {metric: score, 'Frac': frac_tiles}

                print('{0:>6} | {1:>12} | {2:>6} | {3:>6} | {4:.4f} | {5:.4f}'.format(split, smp.idxs[0], y.item(), len(smp), frac_tiles, score))

    smp_statsfile = statsfile.split('.')[0] + '_per_smp.pkl'
    with open(smp_statsfile, 'ab') as f:
        pickle.dump(stats, f)

def calc_overall_stats(datasets, kidx, net, loss_fn, device, gumbel, adv, verbose, statsfile, outfilek, pin_memory, n_workers, metric='AUC'):
    saved_state = torch.load(outfilek, map_location=lambda storage, loc: storage)
    net.load_state_dict(saved_state)

    with open(statsfile, 'rb') as f:
        while 1:
            try:
                stats = pickle.load(f)
                if stats['k'] == kidx and stats['split'] == 'val':
                    best_epoch, best_lamb, best_temp = stats['epoch_best'], stats['lamb_best'], stats['temp_best']
            except EOFError:
                break

    loaders = [DataLoader(d, batch_size=len(d), shuffle=False, pin_memory=pin_memory, num_workers=n_workers, drop_last=False) for d in datasets[1:]]
    
    stats = {}
    for blabel, loader in zip(['val', 'train', 'test'], loaders):
        loss, auc, frac_tiles = trainer.run_validation_loop(0, loader, net, loss_fn, device, lamb=best_lamb, temp=best_temp, gumbel=gumbel, 
            adv=adv, verbose=verbose, blabel=blabel.title())

        stats[blabel] = {metric: auc, 'Loss': loss, 'Frac': frac_tiles}

        print(PRINT_STMT.format(best_epoch, loss, auc, frac_tiles, best_lamb, best_temp, 'Overall', blabel.title(), metric.upper()))

    overall_statsfile = statsfile.split('.')[0] + '_overall.pkl'
    with open(overall_statsfile, 'ab') as f:
        pickle.dump(stats, f)
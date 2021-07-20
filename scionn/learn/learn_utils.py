import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class KLDivLoss(_Loss):
    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, y_pred, y_true):
        y_pred = F.log_softmax(y_pred, dim=-1)
        kldiv = self.loss(y_pred, y_true)
        return kldiv

def calc_per_smp_stats(smps, net, device, gumbel, statsfile, outfile, pin_memory, n_workers, metric='AUC'):
    a = outfile.split('.')
    a[0] = a[0] + '_' + str(kidx).zfill(2)
    outfilek = '.'.join(a)
    
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

    print('{0:>24} | {1:>6} | {2:>6} | {3:>6}'.format('Sample', 'Count', 'Frac', metric))
    
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
                else:
                    y_prob = F.softmax(output_keep.detach(), dim=-1)
                    score = np.mean(np.abs(label.cpu().squeeze(-1).numpy() - y_prob.cpu().squeeze(-1).numpy()))

                stats[split][smp.idxs[0]] = {metric: score, 'Frac': frac_tiles}

                print('{0:>24} | {1:>6} | {2:.4f} | {3:.4f}'.format(smp.idxs[0], len(smp), frac_tiles, score))

    smp_statsfile = statsfile.split('.')[0] + '_per_smp.pkl'
    with open(smp_statsfile, 'ab') as f:
        pickle.dump(stats, f)

def calc_overall_stats(datasets, net, loss_fn, device, gumbel, adv, verbose, statsfile, outfile, pin_memory, n_workers, metric='AUC'):
    a = outfile.split('.')
    a[0] = a[0] + '_' + str(kidx).zfill(2)
    outfilek = '.'.join(a)
    
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

    loaders = [DataLoader(d, batch_size=len(d), shuffle=False, pin_memory=pin_memory, num_workers=n_workers, drop_last=False) for i,d in enumerate(datasets[1:])]
    
    stats = {}
    for blabel, loader in zip(loaders, ['val', 'train', 'test']):
        loss, auc, frac_tiles = trainer.run_validation_loop(0, loader, net, loss_fn, device, lamb=best_lamb, temp=best_temp, gumbel=gumbel, 
            adv=adv, verbose=verbose, blabel=blabel.title())

        stats[blabel] = {metric: auc, 'Loss': loss, 'Frac': frac_tiles}

        print(PRINT_STMT.format(best_epoch, loss, auc, frac_tiles, best_lamb, best_temp, 'Overall', blabel.title(), metric.upper()))

    overall_statsfile = statsfile.split('.')[0] + '_overall.pkl'
    with open(overall_statsfile, 'ab') as f:
        pickle.dump(stats, f)
import scipy
import numpy as np
import anndata as ad
from sklearn.preprocessing import StandardScaler

import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import data_utils
from scionn.models import rnnet, scionnet
from scionn.learn import trainer

MMR_TO_IDX = {
    'MSS': 0,
    'MSI': 1
}

def run_kfold_xvalidation(adata, label, seq_len, batch_size, net_name, net_params, learning_rate, weight_decay, patience, num_epochs, outfile, statsfile, device, kfold=10, 
    ylabel='MMRLabel', ptlabel='PatientBarcode', smlabel='PatientTypeID', training=True, scale=True, trainBase=True, returnBase=True, random_state=32921, verbose=True):

    input_size = adata.shape[1]
    adata.obs[ylabel] = adata.obs[label].apply(lambda x: MMR_TO_IDX[x])

    if random_state is not None:
        np.random.seed(random_state)
        
    idxs_msi = adata.obs.loc[adata.obs[ylabel] == 1, ptlabel].unique()
    idxs_mss = adata.obs.loc[adata.obs[ylabel] == 0, ptlabel].unique()

    if len(idxs_msi) < kfold or len(idxs_mss) < kfold:
        kfold = min(len(idxs_msi), len(idxs_mss))
        print('Insufficient examples, reducing kfold to', kfold)

    np.random.shuffle(idxs_msi)
    np.random.shuffle(idxs_mss)

    splits_msi, splits_mss = np.array_split(idxs_msi, kfold), np.array_split(idxs_mss, kfold)

    for kidx in tqdm(range(kfold)):
        train, val = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, kidx, ptlabel, smlabel, scale=scale, trainBase=trainBase, returnBase=returnBase, train_only=False)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

        if net_name in ['rnnet', 'gru', 'lstm']:
            hidden_size, n_layers, dropout, bidirectional, agg, hide = net_params
            net = rnnet.RNNet(input_size, hidden_size, output_size, n_layers, dropout, bidirectional, agg, hide, net_name=net_name)
            lamb, temp, gumbel, adv = None, None, False, False
        elif net_name == 'scionnet':
            n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, temp, adv, hard, dropout = net_params
            net = SCIONNet(n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, temp, device, adv=adv, hard=hard, dropout=dropout, in_channels=input_size, out_channels=output_size, H_in=seq_len)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, verbose=verbose)

        lamb_orig, temp_orig, best_loss, best_epoch, best_auc, best_fr, tally = lamb, temp, 1e09, 0, 0, 0, 0
        
        for e in tqdm(range(num_epochs)):
            if training:
                trainer.run_training_loop(e, train_loader, net, loss_fn, optimizer, device, lamb=lamb, temp=temp, gumbel=gumbel, adv=adv, verbose=verbose)
            loss, auc, frac_tiles = trainer.rationales_validation_loop(e, val_loader, net, loss_fn, device, lamb=lamb, temp=temp, gumbel=gumbel, adv=adv, verbose=verbose)
            scheduler.step(loss)

            if loss < best_loss: 
                tally = 0
                best_loss, best_epoch, best_auc, best_fr = loss, e, auc, frac_tiles
                print('New best loss:', best_loss, ', New best AUC:', best_auc, ', New best frac:', best_fr)
                torch.save(net.state_dict(), str(kidx).zfill(2) + '_' + outfile)
            else:
                tally += 1

            if tally > patience:
                tally = 0
                saved_state = torch.load(outfile, map_location=lambda storage, loc: storage)
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

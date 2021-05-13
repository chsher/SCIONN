import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients

import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import data_utils
from scionn.models import rnnet, scionnet

MMR_TO_IDX = {
    'MSS': 0,
    'MSI': 1
}

def run_integrated_gradients(adata, label, seq_len, net_name, net_params, outfile, attrfile, device, kfold=10, ylabel='MMRLabel', ptlabel='PatientBarcode', 
    smlabel='PatientTypeID', ctlabel='v11_bot', scale=True, trainBase=True, returnBase=True, random_state=32921, verbose=True):

    net.to(device)

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

    for kidx in range(kfold):
        train, val = data_utils.make_datasets(adata, seq_len, splits_msi, splits_mss, kidx, ptlabel, smlabel, scale=scale, trainBase=trainBase, returnBase=returnBase, train_only=False)
        xb = train.xb

        if net_name in ['rnnet', 'gru', 'lstm']:
            hidden_size, n_layers, dropout, bidirectional, agg, hide = net_params
            net = rnnet.RNNet(input_size, hidden_size, output_size, n_layers, dropout, bidirectional, agg, hide, net_name=net_name)
            lamb, temp, gumbel, adv = None, None, False, False
        elif net_name == 'scionnet':
            n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, temp, adv, hard, dropout = net_params
            net = SCIONNet(n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, temp, device, adv=adv, hard=hard, dropout=dropout, in_channels=input_size, out_channels=output_size, H_in=seq_len)
        
        net.to(device)
        saved_state = torch.load(outfile, map_location=lambda storage, loc: storage)
        net.load_state_dict(saved_state)

        coeffs = pd.DataFrame()

        for dataset in [train, val]:
            for i in tqdm(range(len(dataset))):
                x, y, b = dataset.__getitem__(i)
                input = x.view(1, x.shape[0], x.shape[1])
                baseline = xb.view(1, x.shape[0], x.shape[1])

                igd = IntegratedGradients(net)
                attributions, delta = igd.attribute(input.to(device), baseline.to(device), return_convergence_delta=True)

                new_ids = (np.array(b)).flatten()
                new_ats = attributions.cpu().numpy().reshape((-1, adata.shape[1]))

                df = pd.DataFrame(np.concatenate((new_ids[..., np.newaxis], new_ats), axis=1), columns=['idx'] + list(adata.var.index))
                df['pt'] = df['idx'].apply(lambda x: adata.obs.loc[adata.obs.index[int(x)], smLabel])
                df['cl'] = df['idx'].apply(lambda x: adata.obs.loc[adata.obs.index[int(x)], clLabel])

                coeffs2 = df.iloc[:, 1:].groupby(['cl', 'pt']).mean().fillna(0)
                coeffs3 = coeffs2.swaplevel(i=0, j=1).unstack(1).fillna(0)
                coeffs = pd.concat([coeffs, coeffs3])

        coeffs.to_csv(str(kidx).zfill(2) + '_' + attrfile)
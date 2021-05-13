import scipy
import numpy as np
import anndata as ad
from sklearn.preprocessing import StandardScaler

import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import singlecell

def make_datasets(adata, seq_len, splits_msi, splits_mss, kidx, ptlabel, smlabel, scale=True, trainBase=True, returnBase=True, train_only=False):

    input_size = adata.shape[1]

    if train_only:
        aidxs = np.concatenate((splits_msi, splits_mss))
        data = None
    else:
        bidxs = np.concatenate((splits_msi[kidx], splits_mss[kidx]))
        aidxs = np.concatenate((np.setdiff1d(idxs_msi, bidxs), np.setdiff1d(idxs_mss, bidxs)))

        ddata = adata[adata.obs[ptlabel].isin(bidxs), :]
        bidxs = ddata.obs[smlabel].unique()
        b_labels = {k: ddata.obs.loc[ddata.obs[smlabel] == k, ylabel][0] for k in bidxs}
        b_idxs = {k: np.nonzero(ddata.obs[smlabel].to_numpy() == k)[0] for k in bidxs}

    cdata = adata[adata.obs[ptlabel].isin(aidxs), :]
    aidxs = cdata.obs[smlabel].unique()
    a_labels = {k: cdata.obs.loc[cdata.obs[smlabel] == k, ylabel][0] for k in aidxs}
    a_idxs = {k: np.nonzero(cdata.obs[smlabel].to_numpy() == k)[0] for k in aidxs}
    
    if scale:
        scaler = StandardScaler()
        cdata.X = scipy.sparse.csr_matrix(scaler.fit_transform(cdata.X.todense()))
        
        if ddata is not None:
            ddata.X = scipy.sparse.csr_matrix(scaler.transform(ddata.X.todense()))

        baselineStats = {
            'mean': scaler.mean_,
            'stdev': scaler.scale_
        }

    train = SingleCellDataset(cdata, a_idxs, a_labels, aidxs, len(aidxs), seq_len, input_size, random_state=random_state, baselineStats=baselineStats, trainBaseline=trainBaseline, returnBase=returnBase)

    if train_only:
        return train
    else:
        val = SingleCellDataset(ddata, b_idxs, b_labels, bidxs, len(bidxs), seq_len, input_size, random_state=random_state, baselineStats=baselineStats)
        return train, val
import scipy
import numpy as np
import anndata as ad
from sklearn.preprocessing import StandardScaler

import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import singlecell

def make_splits(adata, ylabel, ptlabel, kfold, random_state=None):
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

    return splits_msi, splits_mss, idxs_msi, idxs_mss

def make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, ylabel, ptlabel, smlabel, scale=True, 
    trainBaseline=True, returnBase=True, details=False, returnTensors=False, train_only=False, random_state=32921):

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
    else:
        baselineStats = {}

    train = singlecell.SingleCellDataset(cdata, a_idxs, a_labels, aidxs, len(aidxs), seq_len, input_size, baselineStats=baselineStats, 
        trainBaseline=trainBaseline, returnBase=returnBase, details=details, random_state=random_state)

    if returnTensors:
        X_train = cdata.X.todense()
        X_test = ddata.X.todense()
        Y_train = cdata.obs.loc[:, ylabel].astype(int)
        Y_test = ddata.obs.loc[:, ylabel].astype(int)

        return X_train, Y_train, X_test, Y_test

    elif train_only:

        return train

    else:
        val = singlecell.SingleCellDataset(ddata, b_idxs, b_labels, bidxs, len(bidxs), seq_len, input_size, baselineStats=baselineStats, 
            trainBaseline=False, returnBase=returnBase, details=details, random_state=random_state)
        
        return train, val
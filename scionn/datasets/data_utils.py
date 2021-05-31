import scipy
import numpy as np
import anndata as ad
from sklearn.preprocessing import StandardScaler

import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.datasets import singlecell

import pdb

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

    return splits_msi, splits_mss, idxs_msi, idxs_mss, kfold

def make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, kfold, ylabel, ptlabel, smlabel, batch_size=None,
    scale=True, trainBaseline=True, returnBase=True, baseOnly=False, details=False, returnTensors=False, bdata=None, random_state=32921):

    input_size = adata.shape[1]

    bidxs = np.concatenate((splits_msi[kidx], splits_mss[kidx]))
    cidxs = np.concatenate((splits_msi[(kidx + 1) % kfold], splits_mss[(kidx + 1) % kfold]))
    aidxs = np.concatenate((np.setdiff1d(idxs_msi, np.concatenate((bidxs, cidxs))), np.setdiff1d(idxs_mss, np.concatenate((bidxs, cidxs)))))

    if bdata is None:
        edata = adata[adata.obs[ptlabel].isin(cidxs), :]
        cidxs = edata.obs[smlabel].unique()
        c_labels = {k: edata.obs.loc[edata.obs[smlabel] == k, ylabel][0] for k in cidxs}
        c_idxs = {k: np.nonzero(edata.obs[smlabel].to_numpy() == k)[0] for k in cidxs}

        ddata = adata[adata.obs[ptlabel].isin(bidxs), :]
        bidxs = ddata.obs[smlabel].unique()
        b_labels = {k: ddata.obs.loc[ddata.obs[smlabel] == k, ylabel][0] for k in bidxs}
        b_idxs = {k: np.nonzero(ddata.obs[smlabel].to_numpy() == k)[0] for k in bidxs}
        
        dataset_sizes = [len(aidxs), len(bidxs), len(cidxs)] if batch_size is None else [batch_size] * 3
    else:
        bidxs = bdata.obs[smlabel].unique()
        b_labels = {k: bdata.obs.loc[bdata.obs[smlabel] == k, ylabel][0] for k in bidxs}
        b_idxs = {k: np.nonzero(bdata.obs[smlabel].to_numpy() == k)[0] for k in bidxs}

        dataset_sizes = [len(aidxs), len(bidxs)] if batch_size is None else [batch_size] * 2

    cdata = adata[adata.obs[ptlabel].isin(aidxs), :]
    aidxs = cdata.obs[smlabel].unique()
    a_labels = {k: cdata.obs.loc[cdata.obs[smlabel] == k, ylabel][0] for k in aidxs}
    a_idxs = {k: np.nonzero(cdata.obs[smlabel].to_numpy() == k)[0] for k in aidxs}
    
    if scale:
        scaler = StandardScaler()
        cdata.X = scipy.sparse.csr_matrix(scaler.fit_transform(cdata.X.todense()))

        if bdata is None:
            ddata.X = scipy.sparse.csr_matrix(scaler.transform(ddata.X.todense()))
            edata.X = scipy.sparse.csr_matrix(scaler.transform(edata.X.todense()))
        else:
            bdata.X = scipy.sparse.csr_matrix(scaler.transform(bdata.X.todense()))

        baselineStats = {
            'mean': scaler.mean_,
            'stdev': scaler.scale_
        }
    else:
        baselineStats = {}

    train = singlecell.SingleCellDataset(cdata, a_idxs, a_labels, aidxs, dataset_sizes[0], seq_len, input_size, baselineStats=baselineStats, 
        trainBaseline=trainBaseline, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)

    if returnTensors:
        if bdata is None:
            X_train, X_val, X_test = cdata.X.todense(), ddata.X.todense(), edata.X.todense()
            Y_train, Y_val, Y_test = cdata.obs.loc[:, ylabel].astype(int), ddata.obs.loc[:, ylabel].astype(int), edata.obs.loc[:, ylabel].astype(int)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test

        else:
            X_train, X_test = cdata.X.todense(), bdata.X.todense()
            Y_train, Y_test = cdata.obs.loc[:, ylabel].astype(int), bdata.obs.loc[:, ylabel].astype(int)
            return X_train, Y_train, X_test, Y_test
        
    else:
        if bdata is None:
            val = singlecell.SingleCellDataset(ddata, b_idxs, b_labels, bidxs, dataset_sizes[1], seq_len, input_size, baselineStats=baselineStats, 
                trainBaseline=False, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)
            test = singlecell.SingleCellDataset(edata, c_idxs, c_labels, cidxs, dataset_sizes[2], seq_len, input_size, baselineStats=baselineStats, 
                trainBaseline=False, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)
            return train, val, test

        else:
            test = singlecell.SingleCellDataset(bdata, b_idxs, b_labels, bidxs, dataset_sizes[1], seq_len, input_size, baselineStats=baselineStats, 
                trainBaseline=False, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)
            return train, test

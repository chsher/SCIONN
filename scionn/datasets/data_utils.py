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

def make_datasets(adata, seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, kfold, ylabel, ptlabel, smlabel, scale=True, 
    trainBaseline=True, returnBase=True, baseOnly=False, details=False, returnTensors=False, return_test=True, bdata=None, random_state=32921):

    input_size = adata.shape[1]

    bidxs = np.concatenate((splits_msi[kidx], splits_mss[kidx]))

    if return_test:
        cidxs = np.concatenate((splits_msi[(kidx + 1) % 10], splits_mss[(kidx + 1) % 10]))
        aidxs = np.concatenate((np.setdiff1d(idxs_msi, np.concatenate((bidxs, cidxs))), np.setdiff1d(idxs_mss, np.concatenate((bidxs, cidxs)))))

        edata = adata[adata.obs[ptlabel].isin(cidxs), :]
        cidxs = edata.obs[smlabel].unique()
        c_labels = {k: edata.obs.loc[edata.obs[smlabel] == k, ylabel][0] for k in cidxs}
        c_idxs = {k: np.nonzero(edata.obs[smlabel].to_numpy() == k)[0] for k in cidxs}    
    else:
        edata = None
        aidxs = np.concatenate((np.setdiff1d(idxs_msi, bidxs), np.setdiff1d(idxs_mss, bidxs)))

    ddata = adata[adata.obs[ptlabel].isin(bidxs), :]
    bidxs = ddata.obs[smlabel].unique()
    b_labels = {k: ddata.obs.loc[ddata.obs[smlabel] == k, ylabel][0] for k in bidxs}
    b_idxs = {k: np.nonzero(ddata.obs[smlabel].to_numpy() == k)[0] for k in bidxs}

    cdata = adata[adata.obs[ptlabel].isin(aidxs), :]
    aidxs = cdata.obs[smlabel].unique()
    a_labels = {k: cdata.obs.loc[cdata.obs[smlabel] == k, ylabel][0] for k in aidxs}
    a_idxs = {k: np.nonzero(cdata.obs[smlabel].to_numpy() == k)[0] for k in aidxs}
    
    if bdata is not None:
        bidxs = bdata.obs[smlabel].unique()
        b_labels = {k: bdata.obs.loc[bdata.obs[smlabel] == k, ylabel][0] for k in bidxs}
        b_idxs = {k: np.nonzero(bdata.obs[smlabel].to_numpy() == k)[0] for k in bidxs}

    if scale:
        scaler = StandardScaler()
        cdata.X = scipy.sparse.csr_matrix(scaler.fit_transform(cdata.X.todense()))
        ddata.X = scipy.sparse.csr_matrix(scaler.transform(ddata.X.todense()))

        if bdata is not None:
            bdata.X = scipy.sparse.csr_matrix(scaler.transform(bdata.X.todense()))

        elif edata is not None:
            edata.X = scipy.sparse.csr_matrix(scaler.transform(edata.X.todense()))

        baselineStats = {
            'mean': scaler.mean_,
            'stdev': scaler.scale_
        }
    else:
        baselineStats = {}

    train = singlecell.SingleCellDataset(cdata, a_idxs, a_labels, aidxs, len(aidxs), seq_len, input_size, baselineStats=baselineStats, 
        trainBaseline=trainBaseline, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)

    if returnTensors:
        if bdata is not None:
            X_train, X_test = cdata.X.todense(), bdata.X.todense()
            Y_train, Y_test = cdata.obs.loc[:, ylabel].astype(int), bdata.obs.loc[:, ylabel].astype(int)
            return X_train, Y_train, X_test, Y_test

        elif edata is not None:
            X_train, X_val, X_test = cdata.X.todense(), ddata.X.todense(), edata.X.todense()
            Y_train, Y_val, Y_test = cdata.obs.loc[:, ylabel].astype(int), ddata.obs.loc[:, ylabel].astype(int), edata.obs.loc[:, ylabel].astype(int)
            return X_train, Y_train, X_val, Y_val, X_test, Y_test

        else:
            X_train, X_val = cdata.X.todense(), ddata.X.todense()
            Y_train, Y_val = cdata.obs.loc[:, ylabel].astype(int), ddata.obs.loc[:, ylabel].astype(int)
            return X_train, Y_train, X_val, Y_val

    else:
        if bdata is not None:
            test = singlecell.SingleCellDataset(bdata, b_idxs, b_labels, bidxs, len(bidxs), seq_len, input_size, baselineStats=baselineStats, 
                trainBaseline=False, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)
            return train, test

        elif edata is not None:
            val = singlecell.SingleCellDataset(ddata, b_idxs, b_labels, bidxs, len(bidxs), seq_len, input_size, baselineStats=baselineStats, 
                trainBaseline=False, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)
            test = singlecell.SingleCellDataset(edata, c_idxs, c_labels, cidxs, len(cidxs), seq_len, input_size, baselineStats=baselineStats, 
                trainBaseline=False, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)
            return train, val, test

        else:
            val = singlecell.SingleCellDataset(ddata, b_idxs, b_labels, bidxs, len(bidxs), seq_len, input_size, baselineStats=baselineStats, 
                trainBaseline=False, returnBase=returnBase, baseOnly=baseOnly, details=details, random_state=random_state)
            return train, val
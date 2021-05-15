import torch
import pickle
import numpy as np
import scanpy as sc
from tqdm.autonotebook import tqdm, trange

import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from scionn.learn import xvaler, attributer
from scionn.datasets import data_utils
from scripts import script_utils

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

MMR_TO_IDX = {
    'MSS': 0,
    'MSI': 1
}

args = script_utils.parse_args()

adata = sc.read_h5ad(args.infile)
adata.obs[args.ylabel] = adata.obs[args.label].apply(lambda x: MMR_TO_IDX[x])

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.net_name in ['logreg', 'rnnet', 'gru', 'lstm', 'scionnet']:
    if args.net_name == 'logreg':
        net_params = [args.output_size, args.dropout] 
    elif args.net_name in ['rnnet', 'gru', 'lstm']:
        net_params = [args.output_size, args.hidden_size[0], args.n_layers, args.dropout, args.bidirectional, args.agg, args.hide] 
    elif args.net_name == 'scionnet':
        net_params = [args.output_size, args.n_conv_layers, args.kernel_size, args.n_conv_filters, args.hidden_size, args.n_layers, args.gumbel, args.lamb, args.temp, args.adv, args.hard, args.dropout]

    if args.training or args.validate:
        xvaler.run_kfold_xvalidation(adata, args.label, args.seq_len, args.batch_size, args.net_name, net_params, args.learning_rate, args.weight_decay, 
            args.patience, args.n_epochs, args.outfile, args.statsfile, device, kfold=args.kfold, ylabel=args.ylabel, ptlabel=args.ptlabel, 
            smlabel=args.smlabel, training=args.training, scale=args.scale, trainBaseline=args.train_baseline, returnBase=args.return_baseline, 
            random_state=args.random_state, verbose=args.verbose)

    if args.attribution:
        attributer.run_integrated_gradients(adata, args.label, args.seq_len, args.net_name, net_params, args.outfile, args.attrfile, device, 
            kfold=args.kfold, ylabel=args.ylabel, ptlabel=args.ptlabel, smlabel=args.smlabel, ctlabel=args.ctlabel, scale=args.scale, 
            trainBaseline=args.train_baseline, returnBase=args.return_baseline, random_state=args.random_state, verbose=args.verbose)

elif args.net_name in ['LR', 'RF']:
    splits_msi, splits_mss, idxs_msi, idxs_mss = data_utils.make_splits(adata, args.ylabel, args.ptlabel, args.kfold, random_state=args.random_state)

    for kidx in trange(args.kfold):
        stats = {'kidx': kidx, 'name': args.net_name, 'auc': {'train': np.nan, 'test': np.nan}}

        X_train, Y_train, X_test, Y_test = data_utils.make_datasets(adata, args.seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, args.ylabel, 
            args.ptlabel, args.smlabel, scale=args.scale, returnTensors=True, random_state=args.random_state)

        if args.net_name == 'LR':
            lr = LogisticRegression(random_state=args.random_state, n_jobs=-1)
        elif args.net_name == 'RF':
            lr = RandomForestClassifier(random_state=args.random_state, n_jobs=-1)

        lr.fit(X_train, Y_train)

        stats['auc']['train'] = roc_auc_score(Y_train, lr.predict_proba(X_train)[:, -1])
        stats['auc']['test'] = roc_auc_score(Y_test, lr.predict_proba(X_test)[:, -1])

        if args.verbose:
            print('k: {0:d}, Train AUC: {1:.4f}, Test AUC: {2:.4f}'.format(kidx, stats['auc']['train'], stats['auc']['test']))

        a = args.outfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        outfilek = '.'.join(a)

        with open(outfilek, 'ab') as f:
            pickle.dump(lr, f)

        with open(args.statsfile, 'ab') as f:
            pickle.dump(stats, f)

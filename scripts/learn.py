import torch
import pickle
import numpy as np
import scanpy as sc
from sklearn.metrics import roc_auc_score
from tqdm.autonotebook import tqdm, trange

import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from scionn.learn import xvaler, attributer
from scionn.models import model_utils
from scionn.datasets import data_utils
from scripts import script_utils

args = script_utils.parse_args()

if args.ylabel == 'MMRLabel':
    MMR_TO_IDX = {
        'MSS': 0,
        'MSI': 1
    }
elif args.ylabel == 'ResponseLabel':
    MMR_TO_IDX = {
        'Non-responder': 0,
        'Responder': 1,
        'PD': 0,
        'SD': 0,
        'PR': 1,
        'CR': 1
    }

adata = sc.read_h5ad(args.infile)
adata.obs[args.ylabel] = adata.obs[args.label].apply(lambda x: MMR_TO_IDX[x])

if args.val_infile is not None:
    bdata = sc.read_h5ad(args.val_infile)
    bdata.obs[args.ylabel] = bdata.obs[args.label].apply(lambda x: MMR_TO_IDX[x])
else:
    bdata = None

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.net_name in ['logreg', 'rnnet', 'gru', 'lstm', 'scionnet']:
    if args.net_name == 'logreg':
        net_params = [args.output_size, args.dropout] 
    elif args.net_name in ['rnnet', 'gru', 'lstm']:
        net_params = [args.output_size, args.hidden_size[0], args.n_layers, args.bidirectional, args.agg, args.hide, args.dropout] 
    elif args.net_name == 'scionnet':
        net_params = [args.output_size, args.n_conv_layers, args.kernel_size, args.n_conv_filters, args.hidden_size, args.n_layers, args.gumbel, args.lamb, args.temp, args.adv, args.hard, args.dropout]

    if args.training or args.validate or bdata is not None:
        xvaler.run_kfold_xvalidation(adata, args.label, args.seq_len, args.batch_size, args.net_name, net_params, args.learning_rate, args.weight_decay, 
            args.patience, args.n_epochs, args.outfile, args.statsfile, device, kfold=args.kfold, ylabel=args.ylabel, ptlabel=args.ptlabel, 
            smlabel=args.smlabel, training=args.training, validate=args.validate, scale=args.scale, trainBaseline=args.train_baseline, returnBase=args.return_baseline, 
            bdata=bdata, pin_memory=args.pin_memory, random_state=args.random_state, verbose=args.verbose)

    if args.train_baseline:
        attributer.check_baseline_training(adata, args.label, args.seq_len, args.batch_size, args.net_name, net_params, args.outfile, args.statsfile, 
            device, kfold=args.kfold, ylabel=args.ylabel, ptlabel=args.ptlabel, smlabel=args.smlabel, scale=args.scale, returnBase=args.return_baseline, 
            bdata=bdata, pin_memory=args.pin_memory, random_state=args.random_state, verbose=args.verbose)

    if args.attribution:
        attributer.run_integrated_gradients(adata, args.label, args.seq_len, args.net_name, net_params, args.outfile, args.statsfile, args.attrfile, 
            device, kfold=args.kfold, ylabel=args.ylabel, ptlabel=args.ptlabel, smlabel=args.smlabel, ctlabel=args.ctlabel, scale=args.scale, 
            trainBaseline=args.train_baseline, returnBase=args.return_baseline, bdata=bdata, random_state=args.random_state, verbose=args.verbose)

        if ('keep' in adata.var.columns) and ('prog_gene' in adata.var.columns) and ('program' in adata.var.columns):
            groundtruth1 = adata.var.loc[(adata.var['keep'] == True) & (adata.var['prog_gene'] == True) & ((adata.var['program'] == 1) | (adata.var['program'] == 3)), :].index
            groundtruth2 = adata.var.loc[(adata.var['keep'] == True) & (adata.var['prog_gene'] == True) & ((adata.var['program'] == 2) | (adata.var['program'] == 3)), :].index
            attributer.compare_groundtruth(args.kfold, groundtruth1, groundtruth2, args.statsfile, args.attrfile, compare_sd=True)

elif args.net_name in ['LR', 'RF']:
    splits_msi, splits_mss, idxs_msi, idxs_mss = data_utils.make_splits(adata, args.ylabel, args.ptlabel, args.kfold, random_state=args.random_state)

    for kidx in trange(args.kfold):
        a = args.outfile.split('.')
        a[0] = a[0] + '_' + str(kidx).zfill(2)
        outfilek = '.'.join(a)

        stats = {'kidx': kidx, 'name': args.net_name, 'auc': {'train': np.nan, 'val': np.nan, 'test': np.nan}}

        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_utils.make_datasets(adata, args.seq_len, splits_msi, splits_mss, idxs_msi, idxs_mss, kidx, args.kfold,
            args.ylabel, args.ptlabel, args.smlabel, scale=args.scale, returnTensors=True, bdata=bdata, random_state=args.random_state)

        lr = model_utils.load_sk_model(args.net_name, outfilek, args.random_state)

        if args.training:
            lr.fit(X_train, Y_train)

        stats['auc']['train'] = roc_auc_score(Y_train, lr.predict_proba(X_train)[:, -1])
        stats['auc']['val'] = roc_auc_score(Y_val, lr.predict_proba(X_val)[:, -1])
        stats['auc']['test'] = roc_auc_score(Y_test, lr.predict_proba(X_test)[:, -1])

        if args.verbose:
            print('k: {0:d}, Train AUC: {1:.4f}, Val AUC: {2:.4f}, Test AUC: {3:.4f}'.format(kidx, stats['auc']['train'], stats['auc']['val'], stats['auc']['test']))

        with open(outfilek, 'ab') as f:
            pickle.dump(lr, f)

        with open(args.statsfile, 'ab') as f:
            pickle.dump(stats, f)

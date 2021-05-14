import torch
import scanpy as sc

import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from scionn.learn import xvaler, attributer
from scripts import script_utils

args = script_utils.parse_args()

adata = sc.read_h5ad(args.infile)

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if args.net_name in ['rnnet', 'gru', 'lstm']:
    net_params = [args.output_size, args.hidden_size[0], args.n_layers, args.dropout, args.bidirectional, args.agg, args.hide] 
elif args.net_name == 'scionnet':
    net_params = [args.output_size, args.n_conv_layers, args.kernel_size, args.n_conv_filters, args.hidden_size, args.n_layers, args.gumbel, args.lamb, args.temp, args.adv, args.hard, args.dropout]

xvaler.run_kfold_xvalidation(adata, args.label, args.seq_len, args.batch_size, args.net_name, net_params, args.learning_rate, args.weight_decay, 
    args.patience, args.n_epochs, args.outfile, args.statsfile, device, kfold=args.kfold, ylabel=args.ylabel, ptlabel=args.ptlabel, smlabel=args.smlabel, 
    training=args.training, scale=args.scale, trainBaseline=args.train_baseline, returnBase=args.return_baseline, random_state=args.random_state, verbose=args.verbose)

if args.attribution:
    attributer.run_integrated_gradients(adata, args.label, args.seq_len, args.net_name, net_params, args.outfile, args.attrfile, device, 
        kfold=args.kfold, ylabel=args.ylabel, ptlabel=args.ptlabel, smlabel=args.smlabel, ctlabel=args.ctlabel, scale=args.scale, 
        trainBaseline=args.train_baseline, returnBase=args.return_baseline, random_state=args.random_state, verbose=args.verbose)

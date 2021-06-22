import torch
import argparse

MMR_FILE = 'drive/My Drive/_data/adata_645k_tsne_20200415_noNAclean_T_chemokinesOnly.h5ad'

def parse_args():
    parser = argparse.ArgumentParser(description='MMR classifier')

    # files
    parser.add_argument('--infile', type=str, default=MMR_FILE, help='file path to anndata object')
    parser.add_argument('--val_infile', type=str, default=None, help='file path to val/test anndata object')
    parser.add_argument('--outfile', type=str, default='drive/My Drive/temp.pt', help='file path to save the model state dict')
    parser.add_argument('--statsfile', type=str, default='drive/My Drive/temp.pkl', help='file path to save the per-epoch stats')
    parser.add_argument('--attrfile', type=str, default='drive/My Drive/temp.csv', help='file path to save the attributions')

    # data
    parser.add_argument('--label', type=str, default='MMRStatus', help='label on which to perform classification task')
    parser.add_argument('--ylabel', type=str, default='MMRLabel', help='binary label on which to perform classification task')
    parser.add_argument('--ptlabel', type=str, default='PatientBarcode', help='patient annotation')
    parser.add_argument('--smlabel', type=str, default='PatientTypeID', help='sample annotation')
    parser.add_argument('--scale', default=False, action='store_true', help='whether or not to standardize the features')
    parser.add_argument('--kfold', type=int, default=10, help='number of train/val/test splits')
    parser.add_argument('--seq_len', type=int, default=100, help='number of cells per sample')
    parser.add_argument('--batch_size', type=int, default=200, help='number of samples per batch')
    parser.add_argument('--pin_memory', default=False, action='store_true', help='whether to pin memory during data loading')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for data loader')
    parser.add_argument('--random_state', type=int, default=31321, help='random seed of the dataset')
    parser.add_argument('--return_baseline', default=False, action='store_true', help='whether or not to return baseline indicator')
    parser.add_argument('--catlabel', type=str, default=None, help='cell category annotation')
    
    # model
    parser.add_argument('--net_name', type=str, default='rnnet', help='name of neural network')
    # both
    parser.add_argument('--dropout', type=float, default=0.5, help='feedforward dropout rate')
    parser.add_argument('--n_layers', type=int, default=2, help='number of fully connected layers')
    parser.add_argument('--output_size', type=int, default=1, help='output size')
    parser.add_argument('--hidden_size', nargs='*', type=int, default=[64, 32], help='hidden size (per layer)')
    # scionn
    parser.add_argument('--kernel_size', nargs='*', type=int, default=[1, 1], help='size of kernels (per layer)')
    parser.add_argument('--n_conv_layers', type=int, default=2, help='number of convolutional layers')
    parser.add_argument('--n_conv_filters', nargs='*', type=int, default=[256, 128], help='number of filters (per layer)')
    parser.add_argument('--adv', default=False, action='store_true', help='whether to train adversarially')
    parser.add_argument('--hard', default=False, action='store_true', help='whether to hard sample')
    parser.add_argument('--lamb', type=float, default=0.0001, help='regularizer weight')
    parser.add_argument('--temp', type=float, default=3.0, help='gumbel-softmax sampling temperature')
    parser.add_argument('--gumbel', default=False, action='store_true', help='whether to gumbel-softmax sample')
    # rnn
    parser.add_argument('--bidirectional', default=False, action='store_true', help='whether to make the rnn bidirectional')
    parser.add_argument('--hide', default=False, action='store_true', help='whether to return the hidden layer')
    parser.add_argument('--agg', default=False, action='store_true', help='whether to sum the embeddings')

    # learning
    parser.add_argument('--training', default=False, action='store_true', help='whether to train the model')
    parser.add_argument('--validate', default=False, action='store_true', help='whether to evaluate the model')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train the model')
    parser.add_argument('--disable_cuda', default=False, action='store_true', help='whether or not to use GPU')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight assigned to L2 regularizer')
    parser.add_argument('--patience', type=int, default=1, help='number of epochs with no improvement before invoking scheduler, model reloading')
    parser.add_argument('--verbose', default=False, action='store_true', help='whether or not to print stats during training')

    # attribution
    parser.add_argument('--train_baseline', default=False, action='store_true', help='whether or not to train the baseline')
    parser.add_argument('--attribution', default=False, action='store_true', help='whether to run integrated gradients')
    parser.add_argument('--ctlabel', type=str, default='v11_bot', help='cell type annotation')
    
    args = parser.parse_args()
    
    return args

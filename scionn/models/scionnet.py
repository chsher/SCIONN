import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from scionn.models import model_utils

class Generator(nn.Module):
    def __init__(self, n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_rnn_layers, dropout=0.5, in_channels=500, out_channels=2, 
        bidirectional=True, num_embeddings=None):
        super(Generator, self).__init__()
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.n_conv_filters = n_conv_filters
        self.hidden_size = hidden_size
        self.n_rnn_layers = n_rnn_layers
        self.conv_layers = []
        self.tanh = nn.Tanh()
        self.num_embeddings = num_embeddings

        if num_embeddings is not None:
            #self.emb = nn.Embedding(num_embeddings, in_channels)
            #in_channels = in_channels * 2
            self.emb = nn.Linear(num_embeddings, in_channels)
        #else:
            #self.emb = nn.Identity()
            #self.emb = None
                 
        for layer in range(self.n_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, self.n_conv_filters[layer], self.kernel_size[layer]))
            self.conv_layers.append(self.tanh)
            in_channels = self.n_conv_filters[layer]
        self.conv = nn.Sequential(*self.conv_layers)

        self.lstm = nn.LSTM(in_channels, self.hidden_size, self.n_rnn_layers, batch_first=True, 
                            dropout=dropout, bidirectional=bidirectional) 
        if bidirectional:
            in_channels = hidden_size * 2
        else:
            in_channels = hidden_size

        self.classification_layer = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        # x: BATCH_SIZE x N_CELLS x N_GENES
        '''if self.emb is None:
            x = torch.transpose(x, 1, 2)

            embed = self.conv(x)
            embed = torch.transpose(embed, 1, 2)

            self.lstm.flatten_parameters()
            output, hidden = self.lstm(embed)
        else:
            output = self.emb(x)'''

        #if self.emb is not None:
        #    embed0 = self.emb(x[:, :, -1].long())
        #    x = torch.cat((x[:, :, :-1], embed0), axis=-1)

        if self.num_embeddings is not None:
            x = x[:, :, -self.num_embeddings:]
            x = self.emb(x)

        x = torch.transpose(x, 1, 2)

        embed = self.conv(x)
        embed = torch.transpose(embed, 1, 2)

        self.lstm.flatten_parameters()
        output, hidden = self.lstm(embed)

        y = self.classification_layer(output)
        return y.squeeze(-1)

class Encoder(nn.Module):
    def __init__(self, n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_fc_layers, dropout=0.5, in_channels=500, out_channels=1, H_in=7,
        num_embeddings=None):
        super(Encoder, self).__init__()
        self.n_conv_layers = n_conv_layers
        self.n_fc_layers = n_fc_layers
        self.kernel_size = kernel_size
        self.n_conv_filters = n_conv_filters
        self.hidden_size = hidden_size
        self.conv_layers = []
        self.fc_layers = []
        self.n = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        #if num_embeddings is not None:
            #self.emb = nn.Embedding(num_embeddings, in_channels)
            #in_channels = in_channels * 2
        #else:
            #self.emb = nn.Identity()
            #self.emb = None
             
        for layer in range(self.n_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, self.n_conv_filters[layer], self.kernel_size[layer]))
            self.conv_layers.append(self.tanh)
            in_channels = self.n_conv_filters[layer]

        in_channels = in_channels * model_utils.update_tile_shape(H_in, self.kernel_size[layer])
        for layer in range(self.n_fc_layers):
            self.fc_layers.append(nn.Linear(in_channels, self.hidden_size[layer]))
            self.fc_layers.append(self.tanh)
            self.fc_layers.append(self.n)
            in_channels = self.hidden_size[layer]

        self.conv = nn.Sequential(*self.conv_layers)
        self.fc = nn.Sequential(*self.fc_layers)
        self.classification_layer = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        # x: BATCH_SIZE x N_CELLS x N_GENES
        
        #if self.emb is not None:
        #    embed0 = self.emb(x[:, :, -1].long())
        #    x = torch.cat((x[:, :, :-1], embed0), axis=-1)

        x = torch.transpose(x, 1, 2)

        embed = self.conv(x)
        embed = embed.view(x.shape[0], -1)

        y = self.fc(embed)
        y = self.classification_layer(y)
        return y

class SCIONNet(nn.Module):
    def __init__(self, n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, gumbel, temp, device, adv=False, hard=False, 
        dropout=0.5, in_channels=500, out_channels=1, H_in=7, bidirectional=True, num_embeddings=None, hide=False):
        super(SCIONNet, self).__init__()

        self.gumbel = gumbel
        self.temp = temp
        self.device = device
        self.adv = adv
        self.hard = hard
        #self.num_embeddings = num_embeddings
        self.hide = hide

        self.gen = Generator(n_conv_layers, kernel_size, n_conv_filters, hidden_size[1], n_layers, in_channels=in_channels, 
            dropout=dropout, bidirectional=bidirectional, num_embeddings=num_embeddings)
        self.enc = Encoder(n_conv_layers, kernel_size, n_conv_filters, hidden_size, n_layers, in_channels=in_channels, 
            out_channels=out_channels, H_in=H_in, num_embeddings=num_embeddings)
    
    def forward(self, x):
        # x: BATCH_SIZE x N_CELLS x N_GENES
        #input = torch.transpose(x, 1, 2)

        '''if self.num_embeddings is not None:
            output = self.gen(x[:, :, -1].long())
        else:
            output = self.gen(x)'''

        output = self.gen(x)

        if self.gumbel:
            keep = model_utils.gumbel_softmax(output, self.temp, self.device, hard=self.hard)
        else:
            if self.hard:
                keep = F.one_hot(torch.argmax(output, dim=-1), output.shape[-1])
            else:
                keep = F.softmax(output, dim=-1)
        x_keep = x.transpose(1, 2).transpose(0, 1) * keep[:, :, -1]
        x_keep = x_keep.transpose(0, 1).transpose(1, 2)

        '''if self.num_embeddings is not None:
            output_keep = self.enc(x_keep[:, :, :-1])
        else:
            output_keep = self.enc(x_keep)'''

        output_keep = self.enc(x_keep)

        if self.adv:
            x_adv = x.transpose(1, 2).transpose(0, 1) * (1 - keep[:, :, -1])
            x_adv = x_adv.transpose(0, 1).transpose(1, 2)
            output_adv = self.enc(x_adv)
            return output_keep, keep, output_adv
        elif not self.hide:
            return output_keep, keep
        else:
            return output_keep

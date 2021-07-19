import torch
import torch.nn as nn

class CNNet(nn.Module):
    def __init__(self, n_conv_layers, kernel_size, n_conv_filters, dropout=0.5, in_channels=500, out_channels=2):
        super(CNNet, self).__init__()

        self.d = torch.nn.Dropout(dropout)

        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.n_conv_filters = n_conv_filters
        self.conv_layers = []
        self.tanh = nn.Tanh()

        for layer in range(self.n_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, self.n_conv_filters[layer], self.kernel_size[layer]))
            self.conv_layers.append(self.tanh)
            in_channels = self.n_conv_filters[layer]
        self.conv = nn.Sequential(*self.conv_layers)

        self.lnr = torch.nn.Linear(in_channels, out_channels)
        
    def forward(self, inputs):
        x = torch.transpose(inputs, 1, 2)

        embed = self.conv(self.d(x))
        embed = torch.transpose(embed, 1, 2)

        y = self.lnr(embed)
        y = y.mean(dim=1)

        return y
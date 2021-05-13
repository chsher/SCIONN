import torch
import torch.nn as nn

class RNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0, bidirectional=False, agg=False, hide=False, net_name='rnnet'):
        super(RNNet, self).__init__()

        self.agg = agg
        self.hide = hide

        if net_name == 'rnnet':
            self.rnn = nn.RNN(input_size=input_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                bias=True, 
                                batch_first=True, 
                                dropout=dropout,
                                bidirectional=bidirectional)
        elif net_name == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                bias=True, 
                                batch_first=True, 
                                dropout=dropout,
                                bidirectional=bidirectional)
        elif net_name == 'gru':
            self.rnn = nn.GRU(input_size=input_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                bias=True, 
                                batch_first=True, 
                                dropout=dropout,
                                bidirectional=bidirectional)
        if bidirectional:
            self.lnr = torch.nn.Linear(hidden_size * 2, output_size)
        else:
            self.lnr = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs, h0=None):
        x, hidden = self.rnn(inputs, h0) # x: batch, seq_len, num_directions * hidden_size
        y = self.lnr(x) # y: batch, seq_len, output_size
        
        if self.agg:
            y = y.sum(dim=1)
        else:
            y = y[:, -1, :]
        
        if self.hide:
            return y
        else:    
            return y, hidden
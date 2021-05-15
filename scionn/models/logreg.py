import torch
import torch.nn as nn

class LogReg(nn.Module):
    def __init__(self, input_size, seq_len, output_size, dropout=0):
        super(LogReg, self).__init__()

        self.d = nn.Dropout(dropout)
        self.lnr = torch.nn.Linear(input_size * seq_len, output_size)
        
    def forward(self, inputs):
        # inputs: batch_size, seq_len, input_size
        batch_size = inputs.shape[0]

        x = inputs.contiguous().view(batch_size, -1)

        y = self.lnr(self.d(x))
        
        return y
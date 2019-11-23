import numpy as np
import torch
import torch.nn as nn

class contentEncoder(nn.Module):
    '''
        A GRU net as the content encoder E_z.
        Parameters:
            Input_dimension: 200, dimension of the word embedding.
            Hidden_dimension: 1000, dimension of the content representation.
            Dropout rate: 0.5
    '''
    def __init__(self, input_dim=200, hidden_dim=1000, drop_rate=0.5):
        super(contentEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, dropout=drop_rate)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h
        
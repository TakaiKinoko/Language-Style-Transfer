import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class styleEncoder(nn.Module):
    '''
    ARCHITECTURE: 
        conv layer -- max pooling -- fully connected layer

    input_dim: 200, dimension of the word embeddings
    hidden_dim: 500, dimension of style representation
    GPU dropout probability: 0.5
    filter size: 200 * {1, 2, 3, 4, 5} with 100 feature maps each
    '''
    def __init__(self, V, D=200, C=2, Ci=1, Co=100, Ks=[1, 2, 3, 4, 5], dropout=0.5):
        super(styleEncoder, self).__init__()
        
        """
        self.args = args
        
        V = args.embed_num      
        D = args.embed_dim    # 200 
        C = args.class_num    # 2 
        Ci = 1
        Co = args.kernel_num  # 100 
        Ks = args.kernel_sizes  # {1, 2, 3, 4, 5}
        """

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
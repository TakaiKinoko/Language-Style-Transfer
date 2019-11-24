import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing import load_data

X_dim = 0
N = 0

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
        '''
        V: number of embeddings (vocabulary size)
        D: embeddings dimension
        C: number of classes
        Ci: number of in_channels
        Co: number of kernels
        Ks: list of kernel sizes 
        '''
        super(styleEncoder, self).__init__()

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

'''
class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = F.sigmoid(x)
        return x
'''

class ContentEncoder(nn.Module):
        '''
        A GRU net as the content encoder E_z.
        Parameters:
            Input_dimension: 200, dimension of the word embedding.
            Hidden_dimension: 1000, dimension of the content representation.
            Dropout rate: 0.5
    '''
    def __init__(self, input_dim=200, hidden_dim=1000, drop_rate=0.5):
        super(ContentEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, dropout=drop_rate)
        #self.lin1 = nn.Linear(X_dim, N)
        

    def forward(self, x):
        out, h = self.gru(x, h)
        return out, h
        #x = self.lin1(x)
        #x = F.dropout(x, p=0.2, training=self.training)
        #x = F.relu(x)
        #x = F.sigmoid(x)
        #return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, x, y):

        return



def train(train_loader):
    num_epochs = 10
    #batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-5

    content_encoder = ContentEncoder().cuda()
    style_encoder = StyleEncoder().cuda()
    generator = Generator().cuda()

    criterion = nn.MSELoss()
    #TODO: consider using different learning rates for each model component
    optimizer = optim.Adam([
                    {'params': latent_content.parameters()},
                    {'params': latent_style.parameters()},
                    {'params': model.parameters()}
                ], lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for sentence, label in train_loader:
            X = Variable(sentence).cuda()
            # ===================forward=====================
            latent_content = content_encoder()
            latent_style = style_encoder()
            output = generator()
            loss = criterion(output, X)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))


if __name__ == "__main__":
    train_loader = load_data()
    train(train_loader)


#
# def train(P, Q, D_gauss, P_decoder, Q_encoder, Q_generator, D_gauss_solver, data_loader):
#     '''
#     Train procedure for one epoch.
#     '''
#     TINY = 1e-15
#     # Set the networks in train mode (apply dropout when needed)
#     ContentEncoder.train()
#     StyleEncoder.train()
#     D_gauss.train()
#
#     # Loop through the labeled and unlabeled dataset getting one batch of samples from each
#     # The batch size has to be a divisor of the size of the dataset or it will return
#     # invalid samples
#     for X, target in data_loader:
#
#         # Load batch and normalize samples to be between 0 and 1
#         X = X * 0.3081 + 0.1307
#         X.resize_(train_batch_size, X_dim)
#         X, target = Variable(X), Variable(target)
#         if cuda:
#             X, target = X.cuda(), target.cuda()
#
#         # Init gradients
#         P.zero_grad()
#         Q.zero_grad()
#         D_gauss.zero_grad()
#
#         #######################
#         # Reconstruction phase
#         #######################
#         z_gauss = Q(X)
#         z_cat = get_categorical(target, n_classes=10)
#         if cuda:
#             z_cat = z_cat.cuda()
#
#         z_sample = torch.cat((z_cat, z_gauss), 1)
#
#         X_sample = P(z_sample)
#         recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)
#
#         recon_loss.backward()
#         P_decoder.step()
#         Q_encoder.step()
#
#         P.zero_grad()
#         Q.zero_grad()
#         D_gauss.zero_grad()

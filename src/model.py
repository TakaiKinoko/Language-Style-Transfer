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

class StyleEncoder(nn.Module):
    '''
    ARCHITECTURE:
        conv layer -- max pooling -- fully connected layer

    input_dim: 200, dimension of the word embeddings
    hidden_dim: 500, dimension of style representation
    GPU dropout probability: 0.5
    filter size: 200 * {1, 2, 3, 4, 5} with 100 feature maps each
    '''

    def __init__(self, D=200, C=2, Ci=1, Co=100, Ks=[1, 2, 3, 4, 5], dropout=0.5):
        '''
        V: number of embeddings (vocabulary size)
        D: embeddings dimension
        C: number of classes
        Ci: number of in_channels
        Co: number of kernels (here: number of feature maps for each kernel size)
        Ks: list of kernel sizes
        dropout: dropout rate
        '''
        super(StyleEncoder, self).__init__()

        # I dont think embedding layer is needed because we used glove vectors as input
        #self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # I dont think embedding layer is needed because we used glove vectors as input
        #x = self.embed(x)  # (N, W, D)

        #if self.args.static:   # baseline model with only a static channel
        #    x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

class ContentEncoder(nn.Module):
    '''
    A GRU net as the content encoder E_z.
    Parameters:
        Input_dimension: 200, dimension of the word embedding.
        Hidden_dimension: 1000, dimension of the content representation.
        Dropout rate: 0.5
    The last hidden state of the GRU E_z is used as the content representation.
    '''
    def __init__(self, input_dim=200, hidden_dim=1000, drop_rate=0.5):
        super(ContentEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, dropout=drop_rate)

    def forward(self, x):
        out, h = self.gru(x, h)
        return out, h


class Generator(nn.Module):
    '''
    A GRU net as the generator G.
    Parameters:
        Input_dimension: 1500, dimension of concat(content_representation, style_representation).
        Hidden_dimension: 200, dimension of the word embedding.
        Dropout rate: 0.5
    The last hidden state of the GRU G is used as the reconstructed word embedding.
    '''
    def __init__(self, input_dim=1500, hidden_dim=200, drop_rate=0.5):
        super(Generator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, dropout=drop_rate)

    def forward(self, x, y):
        out, h = self.gru(x, h)

        #needs to accept 2 inputs, concatenate content and style representations at training time.
        return out, h

class Discriminator(nn.Module):
    def __init__(self, D=200, C=2, Ci=1, Co=250, Ks=[2, 3, 4, 5], dropout=0.5):
        super(Discriminator, self).__init__()
        self.cnn = StyleEncoder(D=D, C=C, Ci=Ci, Co=Co, Ks=Ks, dropout=dropout)

    def forward(self, x):
        x = self.cnn(x)
        x = F.sigmoid(x)
        return x


def train():

    batch_size = 32
    num_epochs = 100
    d_steps = 20
    g_steps = 20

    d_learning_rate = 1e-3
    g_learning_rate = 1e-3
    d_momentum = 0.9
    g_weight_decay = 1e-5

    _, train_loader = load_data(batch_size)

    Ez = ContentEncoder()
    Ey = StyleEncoder()
    G = Generator()
    D = Discriminator()

    criterion = nn.MSELoss()
    #TODO: consider using different learning rates for each model component

    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=d_momentum)

    g_optimizer = optim.Adam([
                    {'params': Ez.parameters()},
                    {'params': Ey.parameters()},
                    {'params': G.parameters()}
                ], lr=g_learning_rate, weight_decay=g_weight_decay)

    print("1")
    for epoch in range(num_epochs):
        for d in range(d_steps):
            print("2")
            D.zero_grad()
            for thing in train_loader:
                print("3")
                print(thing)
                for sentence in sentence_batch:
                    print("4")
                    print(sentence)
                    X = Variable(sentence).cuda()


        for g in range(g_steps):
            for sentence, label in train_loader:
                X = Variable(sentence).cuda()
                # ===================forward=====================
                latent_content = Ez()
                latent_style = Ey()
                output = G()
                loss = criterion(output, X)
                # ===================backward====================
                g_optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))


if __name__ == "__main__":
    train()


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

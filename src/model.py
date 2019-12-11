import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing import load_data, read_glove_vecs, pretrained_embedding_layer, sentences_to_indices

X_dim = 0
N = 0

class EmbeddingLayer(nn.Module):
    '''
    An embedding layer that converts strings to vectors
    Parameters:
        vocab_size: number of embeddings (vocabulary size)
        pretrained_vectors: matrix of word embeddings
        input_dim: embeddings dimension
    '''
    def __init__(self, vocab_size, pretrained_vectors, input_dim=200):
        super(EmbeddingLayer, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_dim)
        pretrained_vectors = np.array(pretrained_vectors)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_vectors))
        self.embed.weight.requires_grad=False

        # self.word_to_index = word_to_index

    def forward(self, word_index):

        # break string up
        # make a size 20 vector
        # for each vector put average
        # for word in
        print("*****************************************")
        print(word_index.shape)
        word_vector = self.embed(word_index)
        return word_vector


class StyleEncoder(nn.Module):
    '''
    ARCHITECTURE:
        conv layer -- max pooling -- fully connected layer

    input_dim: 200, dimension of the word embeddings
    hidden_dim: 500, dimension of style representation
    GPU dropout probability: 0.5
    filter size: 200 * {1, 2, 3, 4, 5} with 100 feature maps each
    '''
    def __init__(self, input_dim=200, C=2, Ci=1, Co=100, Ks=[1, 2, 3, 4, 5], dropout=0.5):
        '''
        input_dim: embeddings dimension
        C: number of classes
        Ci: number of in_channels
        Co: number of kernels (here: number of feature maps for each kernel size)
        Ks: list of kernel sizes
        dropout: dropout rate
        '''
        super(StyleEncoder, self).__init__()

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, input_dim)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
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
    def __init__(self, input_dim=200, hidden_dim=1000, n_layers=1, drop_rate=0.5):
        super(ContentEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, dropout=drop_rate)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class Generator(nn.Module):
    '''
    A GRU net as the generator G.
    Parameters:
        Input_dimension: 1500, dimension of concat(content_representation, style_representation).
        Hidden_dimension: 200, dimension of the word embedding.
        Dropout rate: 0.5
    The last hidden state of the GRU G is used as the reconstructed word embedding.
    '''
    def __init__(self, input_dim=1500, hidden_dim=200, n_layers=1, drop_rate=0.5):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, dropout=drop_rate)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        #needs to accept 2 inputs, concatenate content and style representations at training time.
        return out, h

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class Discriminator(nn.Module):
    def __init__(self, input_dim=200, C=2, Ci=1, Co=250, Ks=[2, 3, 4, 5], dropout=0.5):
        super(Discriminator, self).__init__()
        self.cnn = StyleEncoder(input_dim=input_dim, C=C, Ci=Ci, Co=Co, Ks=Ks, dropout=dropout)

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

    glove_file = '../data/glove.6B/glove.6B.200d.txt'
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)
    vocab_len = len(word_to_index) + 1
    emb_vecs = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    Embed = EmbeddingLayer(vocab_len, emb_vecs)
    Ez = ContentEncoder()
    Ey = StyleEncoder()
    G = Generator()
    D = Discriminator()

    criterion = nn.MSELoss()
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=d_momentum)
    g_optimizer = optim.Adam([
                    {'params': Ez.parameters()},
                    {'params': Ey.parameters()},
                    {'params': G.parameters()}
                ], lr=g_learning_rate, weight_decay=g_weight_decay)

    _, train_loader = load_data(batch_size, word_to_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        for d in range(d_steps):
            d_optimizer.zero_grad()
            for sentence_batch, label_batch in train_loader:
                pass
                #TODO: implement training procedure for the Discriminator

        for g in range(g_steps):
            g_optimizer.zero_grad()
            for sentence_batch, label_batch in train_loader:
                # Ez_h = Ez.init_hidden(batch_size, device)
                # G_h = G.init_hidden(batch_size, device)
                Ez_h = Ez.init_hidden(20, device)
                G_h = G.init_hidden(20, device)

                indices = sentences_to_indices(np.array(sentence_batch), word_to_index, 20)
                X = Variable(torch.from_numpy(indices).long())
                X_vec = Embed(X)

                # ===================DEBUGGING=====================
                print("Hidden--------------------------")
                print(Ez_h.shape)
                print("Input--------------------------")
                print(X_vec.shape)

                # ===================forward=====================
                _, latent_content = Ez(X_vec, Ez_h)
                latent_style = Ey(X_vec)
                _, output = G(torch.cat((latent_content, latent_style), 1), G_h)
                loss = criterion(output, X_vec)
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

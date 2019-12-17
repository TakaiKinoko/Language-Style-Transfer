import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing import load_data, read_glove_vecs, pretrained_embedding_layer, sentences_to_indices
from torch.utils.tensorboard import SummaryWriter


X_dim = 0
N = 0
embedding_dim = 50
style_dim = 50
content_dim = 100
concat_dim = content_dim + style_dim*5
batch_size = 1
sentence_len = 20

class EmbeddingLayer(nn.Module):
    '''
    An embedding layer that converts strings to vectors
    Parameters:
        vocab_size: number of embeddings (vocabulary size)
        pretrained_vectors: matrix of word embeddings
        input_dim: embeddings dimension
    '''
    #def __init__(self, vocab_size, pretrained_vectors, input_dim=200):
    def __init__(self, vocab_size, pretrained_vectors, input_dim=embedding_dim):
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
    #def __init__(self, input_dim=200, C=2, Ci=1, Co=100, Ks=[1, 2, 3, 4, 5], dropout=0.5):
    def __init__(self, input_dim=embedding_dim, C=2, Ci=1, Co=style_dim, Ks=[1, 2, 3, 4, 5], dropout=0.5):
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
        # x = [i.permute(0,2,1) for i in x]

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        #logit = self.fc1(x)  # (N, C)
        return x  #logit


class ContentEncoder(nn.Module):
    '''
    A GRU net as the content encoder E_z.
    Parameters:
        Input_dimension: 200, dimension of the word embedding.
        Hidden_dimension: 1000, dimension of the content representation.
        Dropout rate: 0.5
    The last hidden state of the GRU E_z is used as the content representation.
    '''
    #def __init__(self, input_dim=200, hidden_dim=1000, n_layers=1, drop_rate=0.5):
    def __init__(self, input_dim=embedding_dim, hidden_dim=content_dim, n_layers=1, drop_rate=0.5):
        super(ContentEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, dropout=drop_rate)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out[:,-1,:]

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class Generator(nn.Module):
    '''
    A GRU net as the generator G.
    Parameters:
        #Input_dimension: 1500, dimension of concat(content_representation, style_representation).
        Hidden_dimension: 1500, dimension of concat(content_representation, style_representation).
        #Hidden_dimension: 200, dimension of the word embedding.
        Input_dimension: 200, dimension of the word embedding.
        Dropout rate: 0.5
    The last hidden state of the GRU G is used as the reconstructed word embedding.
    '''
    #def __init__(self, input_dim=1500, hidden_dim=200, n_layers=1, drop_rate=0.5):
    #def __init__(self, input_dim=200, hidden_dim=1500, n_layers=1, drop_rate=0.5):
    def __init__(self, input_dim=concat_dim, hidden_dim=concat_dim, n_layers=1, drop_rate=0.5):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, dropout=drop_rate)
        self.fc = nn.Linear(concat_dim, embedding_dim)

    def forward(self, z, y, start):
        first_hidden =  torch.cat((z, y), 1) # shape: (32, 1500) -- first hidden state

        #for i in range(32):
        # if batch_size != 1:
        #     for i in range(batch_size):
        #         start = torch.stack((start, start))


        #out, h = self.gru(start.view(1, 32, 200), first_hidden.view(1, 32, 1500))
        generated_sentence =[]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_seq = torch.zeros(1, batch_size, sentence_len, concat_dim).to(device)

        input = start.view(1, batch_size, concat_dim)
        hidden = first_hidden.view(1, batch_size, concat_dim)
        for i in range(sentence_len):
            out, h = self.gru(input, hidden)
            input = out
            hidden = h
            output_seq[:,:,i,:] = out

        seq = self.fc(output_seq)
        smaller_output_seq = seq.view(batch_size, sentence_len, embedding_dim)
        return smaller_output_seq

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class Discriminator(nn.Module):
    """
    return 1 if input is from target dataset
    return 0 if input if from source dataset
    """
    # def __init__(self, input_dim=200, C=2, Ci=1, Co=250, Ks=[2, 3, 4, 5], dropout=0.5):
    def __init__(self, input_dim=embedding_dim, C=2, Ci=1, Co=250, Ks=[2, 3, 4, 5], dropout=0.5):
        super(Discriminator, self).__init__()
        self.cnn = StyleEncoder(input_dim=input_dim, C=C, Ci=Ci, Co=Co, Ks=Ks, dropout=dropout)

    def forward(self, x):
        x = self.cnn(x)
        x = F.sigmoid(x)
        return x


def train():
    if torch.cuda.is_available():
        print("CUDA available\n")

    a = torch.cuda.FloatTensor([1.])
    print(a)
    print("\n")

    batch_size = 1
    num_epochs = 100

    d_learning_rate = 1e-3
    g_learning_rate = 1e-3
    y_learning_rate = 1e-3
    d_momentum = 0.9
    g_weight_decay = 1e-5
    y_weight_decay = 1e-5

    glove_file = '../data/glove.6B/glove.6B.' + str(embedding_dim) + 'd.txt'
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)
    vocab_len = len(word_to_index) + 1
    emb_vecs = pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_dim=embedding_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda")

    Embed = EmbeddingLayer(vocab_len, emb_vecs).to(device)
    Ez = ContentEncoder().to(device)
    Ey = StyleEncoder().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)
    D_pretrained = Discriminator().to(device)

    writer = SummaryWriter()
    # writer.add_graph(G)

    criterion = nn.MSELoss()
    d_criterion = nn.BCELoss()
    cycle_criterion = nn.MSELoss()
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=d_momentum)
    g_optimizer = optim.Adam([
                    {'params': Ez.parameters()},
                    {'params': Ey.parameters()},
                    {'params': G.parameters()}
                ], lr=g_learning_rate, weight_decay=g_weight_decay)
    y_optimizer = optim.Adam(Ey.parameters(), lr=y_learning_rate, weight_decay=y_weight_decay)
    p_optimizer = optim.SGD(D_pretrained.parameters(), lr=d_learning_rate, momentum=d_momentum)

    pretrain_loader, train_loader_source, train_loader_target = load_data(batch_size)

    np.random.seed(seed=42)
    #y_target = torch.Tensor(np.random.rand(500))
    y_target = torch.Tensor(np.random.rand(style_dim))
    y_target = y_target.unsqueeze(-1)
    #y_target = y_target.expand(500, 32) #expand to 32 dimensions
    y_target = y_target.expand(style_dim, batch_size) #expand to 32 dimensions
    y_target = y_target.transpose(1, 0).to(device)

    #gen_start_vector = torch.Tensor(np.random.rand(200)).to(device)
    gen_start_vector = torch.Tensor(np.random.rand(concat_dim)).to(device)

    #--------------------------------------
    #TEMPORARY TEST OF RECONSTRUCTION LOSS
    #--------------------------------------
    for epoch in range(num_epochs):
        g_optimizer.zero_grad()
        for sentence_batch, label_batch in train_loader_source:
            Ez_h = Ez.init_hidden(20, device)
            indices = sentences_to_indices(np.array(sentence_batch), word_to_index, 20)
            X = Variable(torch.from_numpy(indices).long()).to(device)
            X_vec = Embed(X).detach()
            # ===================forward=====================
            z = Ez(X_vec, Ez_h)
            y = Ey(X_vec)
            output = G(z, y, gen_start_vector)

            loss = criterion(output, X_vec)
            # ===================backward====================
            g_optimizer.zero_grad()
            loss.backward()
            g_optimizer.step()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))
        writer.add_scalar('loss/train', loss, epoch)
        writer.add_graph(G)



    #**************************************
    #TRAIN STYLE DISCREPANCY DISCRIMINATOR
    #**************************************
    pretrain_epochs = 10
    for epoch in range(pretrain_epochs):
        for sentence_batch, label_batch in pretrain_loader:
            p_optimizer.zero_grad()
            p_decision = D_pretrained(sentence_batch)
            #print(label_batch.shape)
            p_loss = d_criterion(p_decision, Variable(torch.Tensor(label_batch)))
            p_loss.backward()
            p_optimizer.step()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, pretrain_epochs, p_loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))



    for epoch in range(num_epochs):
        for ((source_batch,source_labels),(target_batch,_)) in zip(train_loader_source, train_loader_target):

            #*********************************
            #TRAIN DISCRIMINATOR
            #*********************************
            d_optimizer.zero_grad()

            #TARGET SENTENCES
            #train D to recognize sentences generated from target data
            Ez_h = Ez.init_hidden(20, device)
            G_h = G.init_hidden(20, device)
            indices = sentences_to_indices(np.array(target_batch), word_to_index, 20)
            x_indices = Variable(torch.from_numpy(indices).long())
            x_target = Embed(x_indices).detach()

            z_target = Ez(x_target, Ez_h).detach()
            x_target_gen = G(z_target, y_target, G_h).detach()
            d_target_decision = D(x_target_gen)

            d_target_loss = d_criterion(d_target_decision, Variable(torch.ones([1,1]))) #ones = target
            d_target_loss.backward()


            #SOURCE SENTENCES
            #train D to recognize sentences generated from source data
            Ez_h = Ez.init_hidden(20, device)
            G_h = G.init_hidden(20, device)
            indices = sentences_to_indices(np.array(source_batch), word_to_index, 20)
            x_indices = Variable(torch.from_numpy(indices).long())
            x_source = Embed(x_indices).detach()

            z_source = Ez(x_source, Ez_h).detach()
            x_source_gen = G(z_source, y_target, G_h).detach()
            d_source_decision = D(x_source_gen)

            d_source_loss = d_criterion(d_source_decision, Variable(torch.zeros([1,1]))) #zeros = source
            d_source_loss.backward()

            #update weights
            d_optimizer.step()


            #*********************************
            #TRAIN GENERATOR AND ENCODERS
            #*********************************
            g_optimizer.zero_grad()

            #RECONSTRUCTION LOSS
            #train G to reconstruct the source sentence, given the latent representations
            Ez_h = Ez.init_hidden(20, device)
            G_h = G.init_hidden(20, device)

            indices = sentences_to_indices(np.array(sentence_batch), word_to_index, 20)
            X = Variable(torch.from_numpy(indices).long())
            X_vec = Embed(X)

            z = Ez(X_vec, Ez_h)
            y = Ey(X_vec)
            x_reconstructed = G(z, y, G_h)

            loss_reconstruction = criterion(x_reconstructed, X_vec)
            loss_reconstruction.backward()

            #CYCLE CONSISTENCY LOSS
            z_cycle = Ez(x_reconstructed)
            y_cycle = Ey(x_source)
            x_cycle = G(z_cycle, y_cycle)

            loss_cycle = cycle_criterion(x_cycle, x_source)
            loss_cycle.backward()

            #DISCRIMINATOR LOSS
            #train G to generate sentences from source that fool the discriminator
            d_fake_decision = D(x_source)

            generator_loss = d_criterion(d_fake_decision, Variable(torch,ones([1,1])))
            generator_loss.backward()

            #update weights
            g_optimizer.step()


            #*********************************
            #TRAIN STYLE ENCODER
            #*********************************
            # y_optimizer.zero_grad()
            #
            # style_decision = D_pretrained(sentence)
            # loss_style =
            # loss_style.backward()
            #
            # y_optimizer.step()


        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data[0]))
        if epoch % 10 == 0:
            pic = to_img(output.cpu().data)
            save_image(pic, './mlp_img/image_{}.png'.format(epoch))


if __name__ == "__main__":
    train()

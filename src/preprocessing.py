import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms, utils
import numpy as np


class LabeledSentencesDataset(Dataset):
    def __init__(self, label, text_file, root_dir, transform=None):
        """
        Args:
            label (int): Label to be applied to all samples (0,1)
            text_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label = label

        with open(text_file, "r") as f:
            self.sentences = f.readlines()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentence = self.sentences[idx]

        if self.transform:
            sentence = self.transform(sentence)

        return (sentence, self.label)

    def __len__(self):
        return len(self.sentences)


# class GenerateIndexedSentence(object):
#     """Transform sentence into list of indexes for glove embeddings"""
#     def __init__(self, word_to_index):
#         """
#         args:
#             word_to_index (dict[str]): dictionary of words to index in glove data
#         """
#
#         self.word_to_index = word_to_index
#
#     def __call__(self, sentence):
#         """
#         args:
#             sentence (string): sentence to be converted into list of word embeddings
#         """
#
#         return sentences_to_indices(np.asarray([sentence]), self.word_to_index, 20)
#         # token = self._tokenize(sentence)
#         #
#         # for i,word in enumerate(tokens):
#         #     try:
#         #         vector = self.embeddings_dict[word]
#         #     except:
#         #         print("Word " + word + " not in GloVe dataset")
#
#
# class GenerateWordEmbeddings(object):
#     """Transform sentence into list of word embeddings"""
#
#     def __init__(self, dim, glove_path):
#         """
#         args:
#             dim (int): dimension of word vectors (50, 100, 200, 300)
#         """
#
#         filenames = {
#             50: "glove.6B.50d.txt",
#             100: "glove.6B.100d.txt",
#             200: "glove.6B.200d.txt",
#             300: "glove.6B.300d.txt",
#         }
#
#         self.dim = dim
#         self.padding = [0] * dim # padding -- used in __call__
#
#         self.embeddings_dict = {}
#         with open(glove_path+filenames[dim], 'r', encoding="utf8") as f:
#             for line in f:
#                 values = line.split()
#                 word = values[0]
#                 vector = np.asarray(values[1:], "float32")
#                 self.embeddings_dict[word] = torch.FloatTensor(vector)
#
#     def __call__(self, sample):
#         """
#         args:
#             sample (string): sentence to be converted into list of word embeddings
#         """
#         token_list = self._tokenize(sample)
#
#         sentence_embedding = []
#         for pos in range(20):
#             sentence_embedding = torch.randn(dim)
#
#         for i,word in enumerate(token_list):
#             try:
#                 vector = self.embeddings_dict[word]
#
#                 assert(len(sentence_embedding[i]) == len(vector))
#                 sentence_embedding[i] = vector
#                 #sentence_embedding.append(vector)
#             except:
#                 print("Word " + word + " not in GloVe dataset")
#
#         # while len(sentence_embedding) < 20:  # pad each sentence so they all have length = 20
#         #     sentence_embedding.append(self.padding)
#
#         return sentence_embedding
#
#     def _tokenize(self, string):
#         tokens = string.split(sep=" ")
#         punctuation = [" ", ".", ",", ".\n"]
#
#         tokens_clean = []
#         for t in tokens:
#             t = t.rstrip()
#             t = t.lower()
#             if t not in punctuation:
#                 tokens_clean.append(t)
#
#         return tokens_clean

def sentences_to_indices(X, word_to_index, sentence_len):
    num_sentences = X.shape[0]
    X_indices = np.zeros((num_sentences, sentence_len))

    for i in range(num_sentences):
        sentence_words = X[i].lower().split()

        for j,w in enumerate(sentence_words):
            try:
                X_indices[i, j] = int(word_to_index[w])
            except:
                X_indices[i, j] = int(word_to_index["_UNKNOWN_"])
    return X_indices

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1

    words_to_index["_UNKNOWN_"] = i
    word_to_vec_map["_UNKNOWN_"] = get_average_glove_vector(glove_file)
    return words_to_index, index_to_words, word_to_vec_map

def get_average_glove_vector(glove_file):
    # Get number of vectors and hidden dim
    with open(glove_file, 'r', encoding='UTF-8') as f:
        for i, line in enumerate(f):
            pass
    n_vec = i + 1
    hidden_dim = len(line.split(' ')) - 1

    vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)
    with open(glove_file, 'r', encoding='UTF-8') as f:
        for i, line in enumerate(f):
            vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)

    average_vec = np.mean(vecs, axis=0)
    return average_vec

def pretrained_embedding_layer(word_to_vec_map, word_to_index, emb_dim=200):
    vocab_len = len(word_to_index) + 1  #word index begin with 1,plus 1 for padding 0
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    return emb_matrix

def load_data(batch_size):
    # yelp_train_0 = LabeledSentencesDataset(0, "../data/yelp/sentiment.train.0", "../data/yelp/",
    #                                             transform=GenerateIndexedSentence(word_to_index))
    # yelp_train_1 = LabeledSentencesDataset(1, "../data/yelp/sentiment.train.1", "../data/yelp/",
    #                                             transform=GenerateIndexedSentence(word_to_index))
    yelp_train_0 = LabeledSentencesDataset(0, "../data/yelp/sentiment.train.0", "../data/yelp/")
    yelp_train_1 = LabeledSentencesDataset(1, "../data/yelp/sentiment.train.1", "../data/yelp/")

    full_train = ConcatDataset([yelp_train_0, yelp_train_1])

    size_of_pretrain = 128
    pretrain = Subset(full_train, range(0,size_of_pretrain))
    train = Subset(full_train, range(size_of_pretrain,len(full_train)))

    pretrain_loader = DataLoader(pretrain, batch_size=batch_size, shuffle=True)
    train_loader_source = DataLoader(train, batch_size=batch_size, shuffle=True)
    train_loader_target = DataLoader(yelp_train_1, batch_size=batch_size, shuffle=True)

    return pretrain_loader, train_loader_source, train_loader_target #, dev_loader, test_loader


if __name__ == "__main__":
    load_data()

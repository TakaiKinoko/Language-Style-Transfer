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

        sample = {'sentence': sentence, 'label': self.label}
        return sample

    def __len__(self):
        return len(self.sentences)


class GenerateWordEmbeddings(object):
    """Transform sentence into list of word embeddings"""

    def __init__(self, dim, glove_path):
        """
        args:
            dim (int): dimension of word vectors (50, 100, 200, 300)
        """

        filenames = {
            50: "glove.6B.50d.txt",
            100: "glove.6B.100d.txt",
            200: "glove.6B.200d.txt",
            300: "glove.6B.300d.txt",
        }

        self.padding = [0] * dim # padding -- used in __call__

        self.embeddings_dict = {}
        with open(glove_path+filenames[dim], 'r', encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector

    def __call__(self, sample):
        """
        args:
            sample (string): sentence to be converted into list of word embeddings
        """
        token_list = self._tokenize(sample)

        sentence_embedding = np.zeros(20)
        for i,word in enumerate(token_list):
            try:
                vector = self.embeddings_dict[word]
                sentence_embedding[i] = vector
                #sentence_embedding.append(vector)
            except:
                print("Word " + word + " not in GloVe dataset")

        while len(sentence_embedding) < 20:  # pad each sentence so they all have length = 20
            sentence_embedding.append(self.padding)

        return sentence_embedding

    def _tokenize(self, string):
        tokens = string.split(sep=" ")
        punctuation = [" ", ".", ",", ".\n"]

        tokens_clean = []
        for t in tokens:
            t = t.rstrip()
            t = t.lowercase()
            if t not in punctuation:
                tokens_clean.append(t)

        return tokens_clean

def load_data(batch_size):

    yelp_train_0 = LabeledSentencesDataset(0, "../data/yelp/sentiment.train.0", "../data/yelp/",
                            transform=GenerateWordEmbeddings(200, "../data/glove.6B/"))
    yelp_train_1 = LabeledSentencesDataset(1, "../data/yelp/sentiment.train.1", "../data/yelp/",
                            transform=GenerateWordEmbeddings(200, "../data/glove.6B/"))

    full_train = ConcatDataset([yelp_train_0, yelp_train_1])

    size_of_pretrain = 128
    pretrain = Subset(full_train, range(0,size_of_pretrain))
    train = Subset(full_train, range(size_of_pretrain,len(full_train)))

    pretrain_loader = DataLoader(pretrain, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    return pretrain_loader, train_loader #, dev_loader, test_loader


if __name__ == "__main__":
    load_data()

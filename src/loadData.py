import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np

class sentencesDataset(Dataset):
    def __init__(self, text_file, root_dir, transform=None):
        """
        Args:
            text_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        with open(text_file, "r") as f:
            self.sentences = f.readlines()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.sentences[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


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

        sentence_embedding = []
        for word in token_list:
            try:
                vector = self.embeddings_dict[word]
                sentence_embedding.append(vector)
            except:
                print("Word " + word + " not in GloVe dataset")

        return sentence_embedding

    def _tokenize(self, string):
        tokens = string.split(sep=" ")
        punctuation = [" ", ".", ",", ".\n"]

        tokens_clean = []
        for t in tokens:
            if t not in punctuation:
                tokens_clean.append(t)

        return tokens_clean

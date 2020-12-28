from torch.utils.data import Dataset
from numpy import random
import torch
import pickle

class TextDataset(Dataset):
    def __init__(self, sentence_file, ngram_length):
        print('Loading dataset into memory...')
        self.sentences = pickle.load(open(sentence_file, 'rb'))
        print('Processing dataset...')
        # flatten to give the network a better sense of how blocks of text work
        block = []
        for i in self.sentences:
            for j in i :
                block.append(j)

        self.sentences = block

        self.ngram_length = ngram_length

    def __len__(self):
        return len(self.sentences) - self.ngram_length + 1

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx:(idx + self.ngram_length)])

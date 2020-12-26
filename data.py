from torch.utils.data import Dataset
from numpy import random
import torch
import pickle

class TextDataset(Dataset):
    def __init__(self, sentence_file, ngram_length):
        print('Loading dataset into memory...')
        self.sentences = list(filter(lambda x: True if len(x) > ngram_length else False, pickle.load(open(sentence_file, 'rb'))))

        self.ngram_length = ngram_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        begin = random.randint(len(sentence) - self.ngram_length)
        return torch.tensor(sentence[begin:(begin + self.ngram_length)])

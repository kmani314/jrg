import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim, lmbda, num_cells=1):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # learn a word embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # GRU
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_cells)

        self.drop = nn.Dropout(p=lmbda)

        # fully connected layer at end of net
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        # tensor of word embeddings from input sequence tensor
        embedded = self.embedding(input)
        embedded = self.drop(embedded)
        # embedded = self.drop(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def initHidden(self):
        return torch.zeroes(1, self.hidden_dim)

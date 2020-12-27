import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lmbda, num_cells=1):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # learn a word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_cells)

        self.drop = nn.Dropout(p=lmbda)

        # fully connected layer at end of net
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        # tensor of word embeddings from input sequence tensor
        embedded = self.embedding(input)

        gru_output, gru_hidden = self.gru(embedded)

        output = self.fc(gru_output)
        # output = self.drop(output)
        output2 = self.softmax(output)

        return output2, gru_hidden

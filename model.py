import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lmbda, dropout=0, num_cells=1):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # learn a word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, num_layers=num_cells) # fully connected layer at end of net

        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, cell_state):
        # tensor of word embeddings from input sequence tensor
        embedded = self.embedding(input)

        output, hidden = self.lstm(embedded, (hidden, cell_state))

        output = self.fc1(output)

        output = self.softmax(output)

        return output, hidden

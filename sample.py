import torch
import pickle
from config import hyperparams
from model import RNN
from decode import decode

device = torch.device(hyperparams['device'])
vocab = pickle.load(open('vocab.pkl', 'rb'))

rnn = RNN(
    len(vocab.words),
    hyperparams['embedding_dim'],
    hyperparams['hidden_dim'],
    hyperparams['dropout'],
    num_cells=hyperparams['num_cells'],
    dropout=hyperparams['dropout']
)

state_dict = torch.load('./model.pt')

rnn.load_state_dict(state_dict)
rnn.to(device=device)


def sample(prompt, length):
    return decode(prompt, length, rnn, vocab, device)

import torch
import math
import numpy as np
from config import hyperparams


def decode(prompt, length, rnn, vocab, device):
    hidden = torch.zeros(hyperparams['num_cells'], 1, hyperparams['hidden_dim'], device=device)
    cell_state = torch.zeros(hyperparams['num_cells'], 1, hyperparams['hidden_dim'], device=device)
    aux_idx = vocab.w2i['<UNK>']
    idx = []
    seq = []

    for i in prompt:
        idx.append(vocab.w2i[i])

    this_seq = torch.tensor([idx], device=device).t()

    for i in range(0, length):
        curr_out, (hidden, cell_state) = rnn(this_seq, hidden, cell_state)

        # get last element
        curr_out = torch.narrow(curr_out, 0, len(prompt) - 1, 1)
        curr_out = curr_out.squeeze()
        # remove <SOS>, <EOS>, <UNK>
        curr_out = curr_out[0:aux_idx]

        curr_out = curr_out.tolist()
        curr_out = [math.e**x for x in curr_out]

        # normalize to sum to 1
        sum = np.sum(curr_out)
        curr_out = [x / sum for x in curr_out]

        idx = np.random.choice(len(vocab.words) - 1, p=curr_out)
        idx = torch.tensor([idx], device=device).t()
        this_seq = torch.cat([this_seq[1:], idx.unsqueeze(0)])

        word = vocab.i2w[idx.item()]
        seq.append(word)

    string = ''.join(seq).replace('<SOS>', '').replace('<EOS>', '')

    return '{}{}'.format(''.join(prompt), string)

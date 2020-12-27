import numpy as np
import torch
import pickle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import TextDataset
from vocab import Vocab
from config import hyperparams
from model import RNN

device = torch.device(hyperparams['device'])
vocab = pickle.load(open('vocab.pkl', 'rb'))
writer = SummaryWriter()

rnn = RNN(
    len(vocab.words),
    hyperparams['embedding_dim'],
    hyperparams['hidden_dim'],
    hyperparams['dropout'],
    num_cells=hyperparams['num_cells']
)

rnn.to(device=device)

dataset = TextDataset('./data.pkl', hyperparams['seq_length'])
loader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=0)

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=hyperparams['lr'])
decay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparams['step_size'], gamma=hyperparams['gamma'])

def train(epochs):
    epoch = 1
    for i, minibatch in enumerate(loader):
        hidden = torch.zeros(1, hyperparams['batch_size'], hyperparams['hidden_dim'], device=device)
        if epoch > epochs:
            break

        optimizer.zero_grad()

        gpu_batch = minibatch.to(device=device)
        transpose = torch.transpose(gpu_batch, 0, 1)
        context = torch.narrow(transpose, 0, 0, hyperparams['seq_length'] - 1)


        gt = torch.narrow(transpose, 0, hyperparams['seq_length'] - 1, 1)
        gt = gt.squeeze()

        output, hidden = rnn(context, hidden)

        preds = torch.narrow(output, 0, hyperparams['seq_length'] - 2, 1)

        preds = preds.squeeze()
        loss = criterion(preds, gt)

        loss.backward()

        optimizer.step()
        decay.step()

        print('Epoch {} loss: {}'.format(epoch, loss))

        # if epoch % 100 == 0:
        #     smp = sample('それ', 16)
        #     writer.add_text('Epoch {} sample (Greedy)'.format(epoch), smp)

        writer.add_scalar('Training loss', loss, epoch)
        writer.flush()
        epoch += 1

def sample(prompt, length):
    hidden = torch.zeros(1, 1, hyperparams['hidden_dim'], device=device)
    idx = vocab.w2i[prompt]
    input = torch.tensor([[idx]], device=device)

    output, hidden = rnn(input, hidden)
    output = output.squeeze()
    idx = torch.argmax(output).item()
    return '{}{}'.format(prompt, vocab.i2w[idx])

train(hyperparams['epochs'])
writer.close()

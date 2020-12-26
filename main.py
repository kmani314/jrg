from model import RNN
import pickle
from config import hyperparams
import torch
from torch.utils.data import DataLoader
from data import TextDataset
import numpy as np

cuda = torch.device('cpu')
vocab = pickle.load(open('vocab.pkl', 'rb'))

rnn = RNN(
    len(vocab.words),
    hyperparams['embedding_dim'],
    hyperparams['hidden_dim'],
    hyperparams['dropout'],
    num_cells=hyperparams['num_cells']
)

rnn.to(device=cuda)

dataset = TextDataset('./data.pkl', hyperparams['seq_length'])
loader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=0)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=hyperparams['lr'])
torch.autograd.set_detect_anomaly(True)

def train(epochs):

    epoch = 1
    for i, minibatch in enumerate(loader):
        hidden = torch.zeros(1, hyperparams['batch_size'], hyperparams['hidden_dim'], device=cuda)
        if epoch > epochs:
            break

        optimizer.zero_grad()

        gpu_batch = minibatch.to(device=cuda)
        transpose = torch.transpose(gpu_batch, 0, 1)
        context = torch.narrow(transpose, 0, 0, hyperparams['seq_length'] - 1)
        # out = []
        # for i in context:
        #     out2 = []
        #     for j in i:
        #         out2.append(vocab.i2w[j.item()])
        #     out.append(out2)
        # print(np.array_str(np.array(out)))
        # print(context.shape)


        gt = torch.narrow(transpose, 0, hyperparams['seq_length'] - 1, 1)
        gt = gt.squeeze()

        # out = []
        # for i in gt:
        #     out.append(vocab.i2w[i.item()])
        # print(np.array_str(np.array(out)))
        # print(gt.shape)

        output, hidden = rnn(context, hidden)

        preds = torch.narrow(output, 0, hyperparams['seq_length'] - 2, 1)

        preds = preds.squeeze()

        loss = criterion(preds, gt)

        loss.backward()
        optimizer.step()

        print('Epoch {} loss: {}'.format(epoch, loss))
        epoch += 1

train(hyperparams['epochs'])

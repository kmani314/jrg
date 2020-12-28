import torch
import pickle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import hyperparams
from data import TextDataset
from model import RNN
from util import tokenize_string
from decode import decode

device = torch.device(hyperparams['device'])
vocab = pickle.load(open('vocab.pkl', 'rb'))
writer = SummaryWriter()

rnn = RNN(
    len(vocab.words),
    hyperparams['embedding_dim'],
    hyperparams['hidden_dim'],
    hyperparams['dropout'],
    num_cells=hyperparams['num_cells'],
    dropout=hyperparams['dropout']
)

rnn.to(device=device)

dataset = TextDataset('./data.pkl', hyperparams['seq_length'])
loader = DataLoader(
    dataset,
    batch_size=hyperparams['batch_size'],
    shuffle=True,
    num_workers=0
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=hyperparams['lr'])
decay = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=hyperparams['step_size'],
    gamma=hyperparams['gamma']
)


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(epochs):
    hidden = cell_state = torch.zeros(
        hyperparams['num_cells'],
        hyperparams['batch_size'],
        hyperparams['hidden_dim'],
        device=device
    )

    epoch = 1
    for i, minibatch in tqdm(enumerate(loader), total=epochs):
        hidden = repackage_hidden(hidden)
        cell_state = repackage_hidden(cell_state)
        if epoch > epochs:
            break

        optimizer.zero_grad()

        gpu_batch = minibatch.to(device=device)
        transpose = torch.transpose(gpu_batch, 0, 1)
        context = torch.narrow(transpose, 0, 0, hyperparams['seq_length'] - 1)

        gt = torch.narrow(transpose, 0, hyperparams['seq_length'] - 1, 1)
        gt = gt.squeeze()

        output, (hidden, cell_state) = rnn(context, hidden, cell_state)

        preds = torch.narrow(output, 0, hyperparams['seq_length'] - 2, 1)

        preds = preds.squeeze()
        loss = criterion(preds, gt)

        loss.backward()

        optimizer.step()
        decay.step()

        if epoch % hyperparams['sample_rate'] == 0:
            tokenized = tokenize_string(hyperparams['sample'])
            sample = decode(
                tokenized,
                hyperparams['sample_size'],
                rnn,
                vocab,
                device
            )
            writer.add_text('Greedy decode', sample, global_step=epoch)

        writer.add_scalar('Training loss', loss, epoch)
        writer.flush()
        epoch += 1

    torch.save(rnn.state_dict(), './model.pt')


train(hyperparams['epochs'])
writer.close()

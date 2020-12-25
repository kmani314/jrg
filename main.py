from model import RNN
from util import data_to_ngram
from pprint import pprint
import pickle

data = pickle.load(open('data.pkl', 'rb'))
vocab = pickle.load(open('vocab.pkl', 'rb'))

out = []
for i in data:
    for j in i:
        out.append(j)

# pprint(out[0:5])
res = data_to_ngram(out[0:5], 5, 10)

for i in res:
    print(i)

rnn = RNN(len(vocab.words), 500, 0.6, 5)

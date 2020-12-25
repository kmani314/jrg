from model import RNN
from util import data_to_ngram, tokenize_and_build_vocab
from pprint import pprint
import pickle

# tokenize_and_build_vocab('./data.txt', 50000, 10000)
# data = pickle.load(open('data.pkl', 'rb'))

ngram = pickle.load(open('ngram.pkl', 'rb'))
# print(ngram[0:2])
# res = data_to_ngram(data[0:500000], 8, 10000, processes=12)

# rnn = RNN(len(vocab.words), 500, 0.6, 5)

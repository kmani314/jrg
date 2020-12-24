import re
import nagisa
from data import Vocab
import MeCab
parser = MeCab.Tagger("-O wakati")


# larger, but unsorted
def prune_edict(file, out):
    data = open(file, 'r').read()
    regex = re.compile('^.*?(?=\(|;| \[| /)', re.MULTILINE)
    outfile = open(out, 'a')

    for i in regex.findall(data):
        outfile.write('{}\n'.format(i))
    outfile.close()


# sorted by frequency
def prune_leeds_corpus(file, out, size):
    data = open(file, 'r').read()
    regex = re.compile('(?<=\.\d{2} ).*?$', re.MULTILINE)

    outfile = open(out, 'a')
    for i in regex.findall(data)[0:size]:
        outfile.write('{}\n'.format(i))


def tokenize_jpn(input):
    return nagisa.tagging(input).words


def prune_training_data(file, out, vocab):
    data = open(file, 'r').read().split('。')

    outfile = open(out, 'a')

    max = 0
    for i in data:
        # split = (vocab.w2i.get(x) if vocab.w2i.get(x) is not None
        #          else '<UNK>' for x in nagisa.tagging(i).words)
        split = []
        for j in parser.parse(i).split(' '):
            idx = vocab.w2i.get(j)
            print('{}: {}'.format(j, idx))
            split.append(idx if idx is not None else '<UNK>')

        if len(split) > max:
            max = len(split)

        outfile.write('<SOS>{}<EOS>\n'.format(split))

    return max


# prune_edict('./edict2-utf8.txt', './vocab.txt')
# prune_leeds_corpus('./leeds.txt', './vocab_freq.txt', -1)

words = open('./vocab.txt').read().splitlines()
vocab = Vocab(words)

prune_training_data('./data.txt', './split_data.txt', vocab)

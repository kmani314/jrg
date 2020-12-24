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


def tokenize_and_build_vocab(file, out, vocab_len):
    data = open(file, 'r').read()
    outfile = open(out, 'a')

    words = {}

    tokenized = parser.parse(data)
    for i in tokenized.split(' '):
        if i in words:
            words[i] += 1
            continue
        words[i] = 1

    # sort by freq
    words = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))
    print('{} unique tokens'.format(len(words)))

    # you can get away with much fewer words
    words = [x for x in words][0:vocab_len]
    words.append('<SOS>')
    words.append('<EOS>')
    words.append('<UNK>')

    print(words)
    vocab = Vocab(words)

    out = []
    lines = data.splitlines()
    print(repr(lines))

    for i in lines:
        out2 = []
        out2.append(vocab.w2i.get('<SOS>'))
        for j in parser.parse(i).strip('\n').split(' ')[:-1]:
            idx = vocab.w2i.get(j)
            print(repr('{}: {}'.format(j, idx)))
            out2.append(idx if idx is not None else vocab.w2i.get('<UNK>'))
        out2.append(vocab.w2i.get('<EOS>'))
        out.append(out2)

    for i in out:
        outfile.write('{}\n'.format(i))


# prune_edict('./edict2-utf8.txt', './vocab.txt')
# prune_leeds_corpus('./leeds.txt', './vocab_freq.txt', -1)
tokenize_and_build_vocab('./data.txt', './split_data.txt', 200)

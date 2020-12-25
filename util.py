import re
from data import Vocab
import MeCab
from multiprocessing import Pool
from functools import partial
import pickle
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


def tokenize_block(block):
    count = 0
    words = {}
    tokenized = []
    for i in block:
        tokenized.append(parser.parse(i))

    for j in ' '.join(tokenized).split(' '):
        if j in words:
            # print('{} words in block'.format(count))
            words[j] += 1
            count += 1
            continue
        words[j] = 1
    return (words, tokenized)

def embed_block(vocab, block):
    out = []
    for i in block:

        out2 = []
        out2.append(vocab.w2i.get('<SOS>'))
        for j in i.strip('\n').split(' ')[:-1]:
            idx = vocab.w2i.get(j)
            out2.append(idx if idx is not None else vocab.w2i.get('<UNK>'))
        out2.append(vocab.w2i.get('<EOS>'))
        out.append(out2)

    return out


def tokenize_and_build_vocab(file, vocab_len, chunk):
    data = open(file, 'r').read()

    # chunk data
    print('Chunking into blocks of {} sentences...'.format(chunk))
    lines = data.splitlines()
    blocks = []
    for i in range(0, len(lines), chunk):
        if i % (chunk * 100) == 0:
            print('Blocked {} sentences'.format(i))
        blocks.append(lines[i: i + chunk])

    del data
    pool = Pool(processes=32)

    print('Building vocab over {} blocks...'.format(len(blocks)))
    tokenized = pool.map(tokenize_block, blocks)

    words = {}
    print('Finding unique tokens...')
    for i in [i[0] for i in tokenized]:
        for j in i:
            if j in words:
                words[j] += i[j]
                continue
            words[j] = i[j]

    # sort by freq
    words = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))
    print('{} unique tokens'.format(len(words)))

    # you can get away with much fewer words
    words = [x for x in words][0:vocab_len]
    words.append('<SOS>')
    words.append('<EOS>')
    words.append('<UNK>')
    vocab = Vocab(words)

    p_embed_block = partial(embed_block, vocab)
    res = pool.map(p_embed_block, [i[1] for i in tokenized])

    print('Pickling data')
    pickle.dump(res, open('./data.pkl', 'wb'))
    pickle.dump(vocab, open('./vocab.pkl', 'wb'))


def ngram_block(length, data):
    out = []
    for i in data:
        sent = []
        for j in range(0, len(i) - length, 1):
            sent.append(i[j:j+length])
        out.append(sent)
    return out


def data_to_ngram(data, length, chunk):
    blocks = []
    for i in range(0, len(data), chunk):
        blocks.append(data[i: i + chunk])

    pool = Pool(processes=32)
    p_ngram_block = partial(ngram_block, length)
    res = pool.map(p_ngram_block, blocks)
    print(res)


# prune_edict('./edict2-utf8.txt', './vocab.txt')
# prune_leeds_corpus('./leeds.txt', './vocab_freq.txt', -1)
tokenize_and_build_vocab('./data3.txt', 50000, 10000)

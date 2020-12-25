class Vocab:
    def __init__(self, words):
        self.words = words

        self.w2i = {i: x for x, i in enumerate(words)}
        self.i2w = {x: i for x, i in enumerate(words)}

        print('Vocabulary with {} words constructed.'.format(len(words)))

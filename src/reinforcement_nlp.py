__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'

import numpy as np
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format("trainer/glove.6B.200d.bin")


class LSTM(Chain):
    def __init__(self, in_size, hidden_size, hidden2_size, out_size):
        super(LSTM, self).__init__(
            xh=L.EmbedID(in_size, hidden_size),
            hh=L.LSTM(hidden_size, hidden2_size),
            hy=L.Linear(hidden2_size, out_size),
        )

    def __call__(self, x, t):
        x = Variable(x)
        t = Variable(t)

        h = self.xh(x)
        h = self.hh(h)
        hy = self.hy(h)
        y = F.relu(hy)

        return F.softmax_cross_entropy(y, t)

    def predict(self, x):
        x = Variable(x)

        h = self.xh(x)
        h = self.hh(h)
        hy = self.hy(h)
        y = F.relu(hy)

        y = F.softmax(y)

        return y.data

    def reset(self):
        #self.cleargrads()
        self.hh.reset_state()


def make_pairs(words, max_len=6):
    patterns = defaultdict (list)

    for word in words:
        for second_word in words:
            if word != second_word:
                i = 1
                while (word[:i] == second_word[:i]):
                    i += 1
                if i != 1 and i > max (len (word[i - 1:]), len (second_word[i - 1:])) < max_len:
                    if ("suffix", word[i - 1:], second_word[i - 1:]) in patterns:
                        patterns[("suffix", word[i - 1:], second_word[i - 1:])].append ((word, second_word))
                    else:
                        patterns[("suffix", word[i - 1:], second_word[i - 1:])] = [(word, second_word)]

                i = 1
                while (word[-i:] == second_word[-i:]):
                    i += 1
                if i != 1 and max (len (word[:-i + 1]), len (second_word[:-i + 1])) < max_len:
                    if ("prefix", word[:-i + 1], second_word[:-i + 1]) in patterns:
                        patterns[("prefix", word[:-i + 1], second_word[:-i + 1])].append ((word, second_word))
                    else:
                        patterns[("prefix", word[:-i + 1], second_word[:-i + 1])] = [(word, second_word)]

    return patterns


def molph_classify(thepairs, model, threshold=0.5, min_category=5):
    new_pairs = defaultdict (list)

    for key in thepairs:
        cadidates = thepairs[key]

        similality = []
        for pair in cadidates:
            try:
                cos_sim = model (pair[0], pair[1])
            except:
                pass
            else:
                if cos_sim > threshold:
                    similality.append (pair + (cos_sim,))

        if len (similality) > min_category:
            new_pairs[key] = similality

    return new_pairs


def word2indices(word):
    return np.array([ord(char) - 97 for char in word])


def padding(sentences):
    max_len = np.max([len(s) for s in sentences])
    paded_vec = []
    for sentence in sentences:
        pad_len = max_len - len(sentence)
        pad_vec = [26] * pad_len
        paded_vec.append(np.r_[sentence,pad_vec])

    return np.array(paded_vec,dtype=np.int32)


def make_same_group(pairs, word):
    pair_list = sum (list (pairs.values ()), [])
    group = [pair[0:2] for pair in pair_list if word == (pair[0] or pair[1])]

    return group

def get_hit_rate(pairs,words):
    hit_word = []
    pair_list = sum(list(pairs.values()),[])
    for word in words:
        hit_word.extend([pair[1] for pair in pair_list if word == pair[0]])
        hit_word.extend([pair[0] for pair in pair_list if word == pair[1]])

    c = Counter(hit_word)
    return zip(*c.most_common(80))



if __name__ == '__main__':
    with open ('english_brown.txt') as f:
        data = f.read ()
        data = data.replace('.', '')
        data = data.replace(',', '')
        data = data.replace('""', '')
        data = data.lower()

    all_words = data.split()
    words_set = np.unique(all_words)

    words_set = np.array([word for word in words_set if word.isalpha()])

    model = word_vectors.similarity
    original_pair = make_pairs(words_set, max_len=6)
    pairs = molph_classify(original_pair,model,threshold=0.7,min_category=5)
    word_dic = np.unique(np.ravel([pair[0:2] for pair in sum(list(pairs.values()),[])]))
    common_words,counts = get_hit_rate(pairs,word_dic)


    train_set = [np.unique(np.ravel(make_same_group(pairs,word))) for word in common_words]


    dic_ix = np.array([word2indices(word) for word in word_dic])

    batch_size = 50
    model = LSTM(27,50,20,27)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    loss_record = []
    for i in range(500):
        model.cleargrads()
        sample = np.random.permutation(dic_ix)[:batch_size]
        sample = padding(sample).T
        trainX = sample[:-1]
        trainY = sample[1:]

        loss = 0
        for X,Y in zip(trainX,trainY):
            loss += model(X,Y)

        loss.backward()
        loss_record.append(int(loss.data))
        optimizer.update()


    plt.plot(loss_record)
    plt.show()

    char = np.array([trainX.T[21][0]])
    word = np.apply_along_axis(chr,0, np.array([trainX.T[21] + 97]))
    print(word)

    model.reset()
    while True:
        y = model.predict(char)

        pred = np.apply_along_axis(chr,0, np.argsort(y) + 97)[::-1]
        pred = np.where(pred=='{','end',pred)
        prob = np.sort(y)[0][::-1]
        prob = np.round(prob,5)
        cadidate = {char:p for char,p in zip(pred[:3],prob[:3])}
        print(cadidate)

        char = np.array([np.argmax(y)], dtype=np.int32)

        if char[0] == 26:
            break

    word = word2indices('created').reshape(-1, 1)
    model.reset()
    for char in word:
        y = model.predict(char)

        pred = np.apply_along_axis(chr,0, np.argsort(y) + 97)[::-1]
        pred = np.where(pred=='{','end',pred)
        prob = np.sort(y)[0][::-1]
        prob = np.round(prob,5)
        cadidate = {char:p for char,p in zip(pred[:3],prob[:3])}
        print(cadidate)





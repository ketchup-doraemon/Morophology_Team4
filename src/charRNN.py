__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'


import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L


class Char_Rnn(Chain):
    def __init__(self,vocab_size, in_size, out_size):
        super(Char_Rnn, self).__init__(
            xh=L.EmbedID(vocab_size, in_size,ignore_label=-1),
            hh=L.GRU(in_size, out_size),
        )

    def forward(self, x):
        x = Variable(x)
        x = self.xh(x)
        y = self.hh(x)

        return y

    def __call__(self,word):
        self.reset()

        for char in word:
            out = self.forward(char)

        return out

    def reset(self):
        self.hh.reset_state()

def word_to_index(word):
    word_index = [ord (char) - 96 for char in word]
    word_index.append(0)

    return word_index

def padding(sentences):
    max_len = np.max([len(s) for s in sentences])
    paded_vec = []
    for sentence in sentences:
        pad_len = max_len - len(sentence)
        pad_vec = [-1] * pad_len
        pad_vec.extend(sentence)
        paded_vec.append(pad_vec)

    return np.array(paded_vec,dtype=np.int32)

if __name__ == '__main__':
    with open ('english_brown.txt') as f:
        data = f.read ()
        data = data.replace ('.', '')
        data = data.replace (',', '')
        data = data.replace ('""', '')
        data = data.lower()
    all_words = data.split()
    words_set = np.unique(all_words)[727:]

    word_sample = []
    for word in words_set:
        if word.isalpha():
           word_sample.append(word)

    words_indices = [word_to_index(word) for word in word_sample]



    net = Char_Rnn(27,100,27)


    optimizer = optimizers.Adam()
    optimizer.setup(net)

    batch_size = 50


    loss_record = []
    for i in range(500):

        sample = np.random.permutation(words_indices)
        mini_batch = sample[:batch_size]
        mini_batch = padding(mini_batch)
        loss = 0
        net.cleargrads()

        vec = net(mini_batch.T)
        ans = np.fliplr(mini_batch)[:,1:]
        for correct in ans.T:
            loss += F.softmax_cross_entropy(vec,correct)
            vec = F.argmax(vec,axis=1)

            vec = net.forward(np.array([np.int32(vec.data)]).T)

        loss_record.append(np.float32(loss.data))
        loss.backward()
        optimizer.update()


    else:
        pred = []
        vec = net(mini_batch[0].reshape(-1,1))
        for correct in word:
            vec = F.argmax(vec)
            pred.append(vec.data) 
            vec = net.forward(np.array([np.int32(vec.data)]))


    plt.plot(loss_record)
    plt.show()

    pred = [chr(np.int32(w) + 96) for w in pred]
    print(pred)





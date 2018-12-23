
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L



class First_Network(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(First_Network, self).__init__(
            hh = L.Linear(in_size, hidden_size),
            hy = L.Linear(hidden_size, out_size),
        )

    def __call__(self, x):
        x = Variable(x)

        h = self.hh(x)
        h = F.tanh(h)
        y = self.hy(h)
        y = F.relu(y)

        return F.argmax(y)


class Second_Network(Chain):
    def __init__(self, in_size, hidden_size, hidden2_size, out_size):
        super(Second_Network, self).__init__(
            xh=L.EmbedID(in_size, hidden_size),
            hh=L.LSTM(hidden_size, hidden2_size),
            hh2=L.Linear(hidden2_size, hidden2_size),
            hy=L.Linear(hidden2_size, out_size),
        )

    def __call__(self, x, t):
        x = Variable(x)
        t = Variable(t)

        h = self.xh(x)
        h = F.concat(second_net.xh(x),axis=0).reshape(1,350)
        emb = L.Linear(350,50)
        h = emb(h)

        h = F.dropout(h, 0.1)
        h = F.tanh(h)
        h = self.hh(h)
        h = self.hh2(h)
        h = F.dropout(h, 0.1)
        y = self.hy(h)
        y = F.sigmoid(y)

        return F.mean_squared_error(y,t)

    def predict(self, x):
        x = Variable(x)

        h = self.xh(x)
        h = F.tanh(h)
        h = self.hh(h)
        h = self.hh2(h)
        h = F.relu(self.hy(h))
        y = F.softmax(h)

        return y.data

    def reset(self):
        self.hh.reset_state()


def word2fig(word):
    return np.array([ord(char) - 97 for char in word])


if __name__ == '__main__':
    dic = KeyedVectors.load_word2vec_format("glove/glove.6B.100d.bin")

    with open('english_brown.txt') as f:
        all_txt = f.readlines()
        plain_txt = [sentence.replace('.', '').replace(',', '').replace('""', '').lower().split() for sentence in all_txt]


    original_word = 'worked'
    glove_vec = dic.get_vector(original_word).reshape(1,100)

    first_net = First_Network(100, 100, 5)
    second_net = Second_Network(28, 50, 50, 100)
    optimizer = optimizers.Adam()
    optimizer.setup(first_net)
    optimizer2 = optimizers.Adam()
    optimizer2.setup(second_net)

    first_net.cleargrads()
    second_net.cleargrads()

    loss_record = []
    vecs = []
    for i in range(1000):
        split_ix = first_net(glove_vec)

        vec = [ord(char)-97 for char in original_word]
        vec.insert(int(split_ix.data),27)
        vec = np.array(vec)
        vecs.append(vec)
        second_net.reset()
        loss = second_net(vec,glove_vec)
        loss_record.append(int(loss.data))

        loss.backward()
        optimizer2.update()
        optimizer.update()









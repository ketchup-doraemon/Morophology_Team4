__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'


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
        self.x = x

        h = self.hh(x)
        h = F.tanh(h)
        y = self.hy(h)
        y = F.relu(y)

        return y

class Second_Network(Chain):
    def __init__(self,vocab_size, in_size, hidden_size, out_size):
        super(Second_Network, self).__init__(
            xh=L.EmbedID(vocab_size, in_size),
            hh=L.LSTM(in_size, hidden_size),
            hy=L.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        x = Variable(x)
        x = self.xh(x)
        self.x = x
        h = self.hh(x)
        out = self.hy(h)
        out = F.relu(out)

        return out

    def __call__(self,word):
        self.reset()
        if word == []:
            return self.forward(np.array([0],dtype=np.int32))

        for char in word:
            out = self.forward(char)

        return out

    def reset(self):
        self.hh.reset_state()


class Third_Network(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(Third_Network, self).__init__(
            hh = L.Linear(in_size, hidden_size),
            hy = L.Linear(hidden_size, out_size),
        )

    def __call__(self, x,t):
        t = Variable(t)
        self.x = x

        h = self.hh(x)
        h = F.relu(h)
        out = self.hy(h)
        out = F.tanh(out)

        return F.mean_squared_error(out,t)


if __name__ == '__main__':
    dic = KeyedVectors.load_word2vec_format("trainer/glove.6B.100d.bin")
    original_word = 'worked'
    glove_vec = dic.get_vector(original_word).reshape(1, 100)


    first_net = First_Network(100, 30, len(original_word))
    optimizer1 = optimizers.Adam()
    optimizer1.setup(first_net)
    second_net = Second_Network(27, 3, 10, 50)
    second_net.reset()
    optimizer2 = optimizers.Adam()
    optimizer2.setup(second_net)
    second_net2 = Second_Network(27, 3, 10, 50)
    second_net2.reset()
    optimizer3 = optimizers.Adam()
    optimizer3.setup(second_net2)
    third_net = Third_Network(100, 100, 100)
    optimizer4 = optimizers.Adam()
    optimizer4.setup(third_net)


    loss_record = []
    for i in range(1):
        f1 = first_net(glove_vec)
        split_ix = np.argmax(f1.data, axis=1) + 1

        word1 = original_word[:np.int(split_ix)]
        word2 = original_word[np.int(split_ix):]

        vec1 = [np.array([ord(char) - 96], dtype=np.int32) for char in word1]
        y1 = second_net(vec1)

        vec2 = [np.array([ord(char) - 96], dtype=np.int32) for char in word2]
        y2 = second_net2(vec2)

        concated = F.concat([y1, y2])

        loss = third_net(concated, glove_vec)
        loss_record.append(loss.data)
        loss.backward(retain_grad=True)
        optimizer4.update()
        optimizer3.update()
        optimizer2.update()

        f1_loss = F.concat([second_net.x.grad,second_net2.x.grad])
        f1.grad = f1_loss.data
        f1.backward(retain_grad=True)
        optimizer1.update()














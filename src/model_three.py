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
            bn = L.BatchNormalization(hidden_size),
            hy = L.Linear(hidden_size, out_size),
            bn2 = L.BatchNormalization(out_size)
        )

    def __call__(self, x):
        x = Variable(x)
        h = self.hh(x)
        #h = self.bn(h)
        h = F.dropout(h,0.3)
        h = F.relu(h)

        y = self.hy(h)
        #y = self.bn2(y)
        y = F.softmax(y)

        return y


class Second_Network(Chain):
    def __init__(self,vocab_size, in_size, out_size):
        super(Second_Network, self).__init__(
            xh=L.EmbedID(vocab_size, in_size),
            hh=L.LSTM(in_size, out_size),
        )

    def forward(self, x):
        x = Variable(x)
        x = self.xh(x)
        if self.i == 0:
            self.x = x
        y = self.hh(x)

        return y

    def __call__(self,word):
        self.reset()
        self.i = 0
        if word == []:
            return self.forward(np.array([0],dtype=np.int32))
        
        for char in word:
            out = self.forward(char)
            self.i += 1

        return out

    def reset(self):
        self.hh.reset_state()


class Third_Network(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(Third_Network, self).__init__(
            hh1 = L.Linear(in_size, hidden_size),
            bn1 = L.BatchNormalization(hidden_size),
            hh2 =L.Linear(in_size, hidden_size),
            bn2 = L.BatchNormalization(hidden_size),
            hy = L.Linear(hidden_size*2, out_size),
            bn3 = L.BatchNormalization(out_size),
        )

    def __call__(self, x1,x2,t):
        t = Variable(t)
        h1 = self.hh1(x1)
        h2 = self.hh2(x2)
        #h1 = self.bn1(h1)
        #h2 = self.bn2(h2)
        h1 = F.dropout(h1,0.3)
        h2 = F.dropout(h2, 0.3)
        h1 = F.relu(h1)
        h2 = F.relu(h2)
        h = F.concat([h1, h2])
        out = self.hy(h)
        #out = self.bn3(out)
        #out = F.dropout(out,0.3)
        out = F.tanh(out)
        out = F.normalize(out)

        return F.mean_squared_error(out,t)


if __name__ == '__main__':
    dic = KeyedVectors.load_word2vec_format("trainer/glove.6B.100d.bin")
    original_word = 'does'
    glove_vec = dic.get_vector(original_word).reshape(1, 100)


    first_net = First_Network(100,30,len(original_word))
    first_net.cleargrads()
    optimizer1 = optimizers.Adam()
    optimizer1.setup(first_net)
    second_net = Second_Network(27,30, 50)
    second_net.cleargrads()
    second_net.reset()
    optimizer2 = optimizers.Adam()
    optimizer2.setup(second_net)
    second_net2 = Second_Network(27, 30, 50)
    second_net2.cleargrads()
    second_net2.reset()
    optimizer3 = optimizers.Adam()
    optimizer3.setup(second_net2)
    third_net = Third_Network(50, 50, 100)
    third_net.cleargrads()
    optimizer4 = optimizers.Adam()
    optimizer4.setup(third_net)


    loss_record = []
    f1_record = []
    for i in range(50):
        f1 = first_net(glove_vec)
        split_ix = np.argmax(f1.data,axis=1) + 1

        word1 = original_word[:np.int(split_ix)]
        word2 = original_word[np.int(split_ix):]

        vec1 = [np.array([ord(char) - 96], dtype=np.int32) for char in word1]
        y1 = second_net(vec1)

        vec2 = [np.array([ord(char) - 96], dtype=np.int32) for char in word2]
        y2 = second_net2(vec2)


        loss = third_net(y1,y2,glove_vec)
        loss_record.append(float(loss.data))
        loss.backward(retain_grad=True)


        optimizer4.update()
        optimizer3.update()
        optimizer2.update()

        f1_loss = F.concat([second_net.x.grad,second_net2.x.grad])
        f1_loss = F.sum(F.absolute(f1_loss))
        f1_record.append(float(f1_loss.data))
        f1.grad = (f1 * f1_loss.data).data
        #f1.grad = (f1 * loss.data).data
        f1.unchain_backward()
        f1.backward(retain_grad=True)
        optimizer1.update()

    plt.plot(loss_record)
    plt.show()
    plt.plot(f1_record)
    plt.show()

    print(original_word,'is :')
    print(word1,'+',word2)







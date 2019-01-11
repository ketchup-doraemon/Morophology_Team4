# -*- coding: utf-8 -*-
__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L

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
    
    
    def predict(self, x1,x2):
        h1 = self.hh1(x1)
        h2 = self.hh2(x2)
        h1 = F.relu(h1)
        h2 = F.relu(h2)
        h = F.concat([h1, h2])
        out = self.hy(h)
        out = F.tanh(out)
        out = F.normalize(out)

        return out


if __name__ == '__main__':
    dic = KeyedVectors.load_word2vec_format("trainer/glove.6B.100d.bin")
    original_word = 'created'
    glove_vec = dic.get_vector(original_word).reshape(1, 100)


    second_net = Second_Network(27,100, 50)
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
    word1 = 'work'
    word2 = 'ed'
    vec1 = [np.array([ord(char) - 96], dtype=np.int32) for char in word1]
    vec2 = [np.array([ord(char) - 96], dtype=np.int32) for char in word2]
    for i in range(100):

        y1 = second_net(vec1)
        y2 = second_net2(vec2)

        loss = third_net(y1,y2,glove_vec)
        loss_record.append(float(loss.data))
        loss.backward(retain_grad=True)

        optimizer4.update()
        optimizer3.update()
        optimizer2.update()
        
        
            
    y1 = second_net(vec1)
    y2 = second_net2(vec2)
    pred = third_net.predict(y1,y2)
    
    dic.most_similar(pred.data)
    
    plt.plot(loss_record)
    plt.show()
    
    print(dic.most_similar(pred.data))

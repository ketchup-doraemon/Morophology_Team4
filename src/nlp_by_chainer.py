# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:57:26 2018

@author: dy248
"""

import numpy as np
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L


with open('english_brown.txt') as f:
    data = f.read()
    data = data.replace('.','')
    data = data.replace(',','')
    data = data.replace('""','')
    data = data.lower()
    
all_words = data.split()
words_set = np.unique(all_words)[727:]


def one_hot_expresssion(word):
    n_class = 26
    fig_word = [ord(char) - 97 for char in word]

    return np.identity(n_class)[fig_word]

def softmax(array):
    return np.exp(array)/np.sum(np.exp(array))


class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(LSTM, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh = L.LSTM(hidden_size, hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
 
    def __call__(self, x, t=None, train=False):
        x = Variable(x)
        if train:
            t = Variable(t)
        h = self.xh(x)
        h = self.hh(h)
        y = self.hy(h)
        if train:
            return F.mean_squared_error(y, t)
        else:
            return y.data
 
    def reset(self):
        # 勾配の初期化とメモリの初期化
        self.zerograds()
        self.hh.reset_state()

model = LSTM(in_size=26, hidden_size=100, out_size=26)
optimizer = optimizers.Adam()
optimizer.setup(model)

trainX = one_hot_expresssion('working').astype(np.float32)


for i in range(2000):
    model.reset()
    loss = 0
    for i in range(trainX.shape[0] - 1):    
        x = trainX[i].reshape(1,26)
        t = trainX[i+1].reshape(1,26)
        loss += model(x=x, t=t, train=True)
         
    loss.backward()
    loss.unchain_backward()
    optimizer.update()
    
model.reset()
print('w :', 1.)
for X in trainX[:-1]:
    y = model(X.reshape(1,26),train=False)
    prob = np.max(softmax(y))
    y = chr(np.argmax(y) + 97)
    
    print(y +' :',prob)


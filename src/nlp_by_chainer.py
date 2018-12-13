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


def one_hot_expression(word):
    
    fig_word = [ord(char) - 97 for char in word]
    
    return np.array(fig_word)
    #n_class = 26
    #return np.identity(n_class)[fig_word]


def softmax(array):
    return np.exp(array)/np.sum(np.exp(array))


class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(LSTM, self).__init__(
            xh = L.EmbedID(in_size, hidden_size),
            #xh = L.Linear(in_size, hidden_size),
            hh = L.LSTM(hidden_size, hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
 
    def __call__(self, x, t=None, train=False):
        x = Variable(x)
        if train:
            t = Variable(t)
        h = self.xh(x)
        h = self.hh(h)
        y = F.softmax(self.hy(h))
        #y = F.max(y)
        if train:
            return F.softmax_cross_entropy(y, t)
        else:
            return y.data
 
    def reset(self):
        self.cleargrads()
        self.hh.reset_state()
        

model = LSTM(in_size=26, hidden_size=100, out_size=26)

optimizer = optimizers.Adam()
optimizer.setup(model)

trainX = one_hot_expression('worker').astype(np.int32)
trainY = one_hot_expression('working').astype(np.int32)
trainZ = one_hot_expression('workshop').astype(np.int32)
trainX2 = one_hot_expression('works').astype(np.int32)
trainY2 = one_hot_expression('worked').astype(np.int32)
trainZ2 = one_hot_expression('work').astype(np.int32)


for i in range(1000):
    model.reset()
    loss = 0
    for i in range(trainX.shape[0] - 1):    
        x = np.array([trainX[i]])
        t = np.array([trainX[i+1]])
        loss += model(x=x, t=t, train=True)
         
    loss.backward()
    #loss.unchain_backward()
    optimizer.update()
    
    
    model.reset()
    loss = 0
    for i in range(trainY.shape[0] - 1):    
        x = np.array([trainY[i]])
        t = np.array([trainY[i+1]])
        loss += model(x=x, t=t, train=True)
         
    loss.backward()
    #loss.unchain_backward()
    optimizer.update()
    
    model.reset()
    loss = 0
    for i in range(trainZ.shape[0] - 1):    
        x = np.array([trainZ[i]])
        t = np.array([trainZ[i+1]])
        loss += model(x=x, t=t, train=True)
         
    loss.backward()
    #loss.unchain_backward()
    optimizer.update()
    
    model.reset()
    loss = 0
    for i in range(trainX2.shape[0] - 1):    
        x = np.array([trainX2[i]])
        t = np.array([trainX2[i+1]])
        loss += model(x=x, t=t, train=True)
         
    loss.backward()
    #loss.unchain_backward()
    optimizer.update()
    
    
    model.reset()
    loss = 0
    for i in range(trainY2.shape[0] - 1):    
        x = np.array([trainY2[i]])
        t = np.array([trainY2[i+1]])
        loss += model(x=x, t=t, train=True)
         
    loss.backward()
    #loss.unchain_backward()
    optimizer.update()



print('w :', 1.)

char = np.array([trainX[0]])
model.reset()
for i in range(5):
    y = model(char,train=False)
    prob = np.max(y)
    char = np.array([np.argmax(y)],dtype=np.int)
    y = chr(np.argmax(y) + 97)
    
    print(y +' :',prob)    
    
    
    
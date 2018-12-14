# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:57:26 2018

@author: dy248
"""

import numpy as np
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L


#The function converts the word to a figure.
def word2fig(word):   
    return np.array([ord(char) - 97 for char in word])


class LSTM(Chain):
    def __init__(self, in_size, hidden_size,hidden2_size, out_size):
        super(LSTM, self).__init__(
            xh = L.EmbedID(in_size, hidden_size),
            bn = L.BatchNormalization(hidden_size),
            hh = L.LSTM(hidden_size, hidden2_size),
            hh2 = L.LSTM(hidden2_size, hidden2_size),
            hy = L.Linear(hidden2_size, out_size),         
            )
 
    def __call__(self, x, t):
        x = Variable(x)
        t = Variable(t)
        
        h = self.xh(x)
        h = self.bn(h)
        h = F.tanh(h)
        h = self.hh(h)
        h = self.hh2(h)
        y = F.softmax(self.hy(h))
        
        return F.softmax_cross_entropy(y, t)

    def predict(self,x):
        x = Variable(x)
        
        h = self.xh(x)
        h = self.bn(h)
        h = F.tanh(h)
        h = self.hh(h)
        h = self.hh2(h)
        y = F.softmax(self.hy(h))
        
        return y.data        
 
    def reset(self):
        self.cleargrads()
        self.hh.reset_state()
        
        
if __name__ == '__main__':
    
    with open('english_brown.txt') as f:
        data = f.read()
        data = data.replace('.','')
        data = data.replace(',','')
        data = data.replace('""','')
        data = data.lower()
        
    all_words = data.split()
    words_set = np.unique(all_words)[727:]
    

    model = LSTM(27,300,300,27)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    training_data = ['worker','working','workshop','works','worked','work','workers']
    training_data = [np.hstack([word2fig(x),26]).astype(np.int32) for x in training_data]
    
    loss_record = []
    for i in range(1000):
        model.reset()
        trainX = np.random.choice(training_data,1)[0]
        loss = 0
        for i in range(trainX.shape[0] - 1):    
            x = np.array([trainX[i]])
            t = np.array([trainX[i+1]])
            loss += model(x=x, t=t)
             
        loss_record.append(int(loss.data))
        loss.backward()
        optimizer.update()
        

    print('w :', 1.)
    
    char = np.array([trainX[0]])
    model.reset()
    for i in range(10):
        y = model.predict(char)
        prob = np.max(y)
        char = np.array([np.argmax(y)],dtype=np.int)
        y = chr(np.argmax(y) + 97)
        
        print(y +' :',prob)    
        
    import pylab as plt
    plt.plot(loss_record)
    
    
    
    
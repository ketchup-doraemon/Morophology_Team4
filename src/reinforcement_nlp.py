__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'


import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from copy import deepcopy

from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L


df = pd.read_csv('trainer/occurrence_probability.csv',index_col=0)
def reward(word):
    return df[word]*[1.01**x for x in range(df.shape[0])]
    
def one_hot_encoding(indices,n_class=27):
    emb_zero = (indices != 26).astype(int)
    
    return emb_zero.reshape(-1,1)*np.eye(n_class)[indices]


def word_to_index(word):
    word_index = [ord (char) - 97 for char in word]
    return word_index


def normalize(data):
    vmax = np.nanmax(data)
    vmin = np.nanmin(data)
    
    return (data-vmin)/(vmax-vmin)


def mutation(_param):
    new_param = []
    for param in _param:
        shape = np.shape(param)
        i = random.randrange(shape[0])
        j = random.randrange(shape[1])
             
        param[i,j] = param[i,j] + random.uniform(-0.1,0.1)
           
        new_param.append(param)
        
    return  new_param 

def padding(sentences):
    max_len = np.max([len(s) for s in sentences])
    paded_vec = []
    for sentence in sentences:
        pad_len = max_len - len(sentence)
        pad_vec = [26] * pad_len
        pad_vec.extend(sentence)
        paded_vec.append(pad_vec)

    return np.array(paded_vec,dtype=np.int32)
   
    
class LSTM(Chain):
    def __init__(self, in_size, hidden_size,hidden2_size,out_size):
        super(LSTM, self).__init__(
            xh=L.Linear(in_size, hidden_size),
            hh=L.LSTM(hidden_size, hidden2_size),
            hh2=L.LSTM(hidden2_size, hidden2_size),
            hy=L.Linear(hidden2_size,out_size))

    def forward(self, x):
        x = Variable(x)
        h = self.xh(x)
        h = F.dropout(h,0.3)
        h = F.relu(h)
        h2 = self.hh(h)
        h = F.dropout(h2, 0.3)
        h2 = self.hh2(h2)
        y = self.hy(h2)

        return y
    
    def __call__(self,word,ans):
        self.reset()
        for char in word:
            res = self.forward(char)
            
        return F.softmax_cross_entropy(res,ans,reduce='mean')

    def reset(self):
        self.hh.reset_state()
        self.hh2.reset_state()
        
        
if __name__ == '__main__':
    reward_matrix = (df.T*[1.01**x for x in range(df.shape[0])])
    answer = np.nanargmax(reward_matrix.values,axis=1).astype(np.int32)
    training_data = [word_to_index(x) for x in df.columns]
    training_data = padding(training_data).T
    training_data = [one_hot_encoding(x).astype(np.float32) for x in training_data]
    model = LSTM(27,50,20,27)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    loss_record = []
    for i in range(1500):
        model.cleargrads()
        #training_sample = np.random.permutation(training_data)
        loss = model(training_data,answer)
        loss_record.append(np.float(loss.data))
        
        loss.backward()
        optimizer.update()
    
    plt.plot(loss_record)


#np.vstack([padding(training_data).T,answer])

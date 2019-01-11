__author__ = 'Daisuke Yoda'
__Date__ = 'January 2019'


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from chainer import Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L

def word_to_index(word):
    word_index = [ord (char) - 97 for char in word]
    return word_index


def one_hot_encoding(indices, n_class=27):
    return np.eye(n_class)[indices]

def padding(sentences):
    max_len = np.max([len(s) for s in sentences])
    paded_vec = []
    for sentence in sentences:
        pad_len = max_len - len(sentence)
        pad_vec = [-1] * pad_len
        pad_vec.extend(sentence)
        paded_vec.append(pad_vec)

    return np.array(paded_vec, dtype=np.int32)


class LSTM(Chain):
    def __init__(self,vocab_size,in_size,hidden_size,hidden2_size, out_size):
        super(LSTM, self).__init__(
            xh=L.EmbedID(vocab_size, in_size,ignore_label=-1),
            hh=L.LSTM(in_size, hidden_size),
            hh2=L.LSTM(hidden_size, hidden2_size),
            hy=L.Linear(hidden2_size,out_size))

    def __call__(self,input_x,t):
        self.reset()
        for x in input_x:
            x = Variable(x)
            x = self.xh(x)
            x = F.tanh(x)
            x = F.dropout(x,0.3)
            h = self.hh(x)
            h = self.hh2(h)
            h = F.dropout(h,0.3)

        else:
            y = self.hy(h)
            y = F.relu(y)

        #return F.softmax_cross_entropy(y,t)
        return F.mean_squared_error(y,t)

    def predict(self,input_x):
        self.reset()
        for x in input_x:
            x = Variable(x)
            x = self.xh(x)
            x = F.tanh(x)
            h = self.hh(x)
            h = self.hh2 (h)

        else:
            y = self.hy(h)
            y = F.relu(y)

        return np.argmax(y.data,axis=1)


    def reset(self):
        self.hh.reset_state()



if __name__ == '__main__':
    df = pd.read_csv('trainer/split_point.csv',index_col=0)
    original_data = [word_to_index(x) for x in df.columns]
    original_data = padding(original_data)

    split_point = np.argmax(np.nan_to_num(df,0),axis=0)
    split_point = np.nan_to_num(df,0).T
    model = LSTM(27,10,30,15,split_point.shape[1])
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    trainX = original_data[:500]
    trainY = split_point[:500]
    testX = original_data[500:]
    testY = split_point[500:]

    trainX = trainX.T.astype(np.int32)
    testX = testX.T.astype(np.int32)
    trainY = np.array(trainY,dtype=np.float32)
    testY = np.array(testY,dtype=np.float32)

    train_loss = []
    test_loss = []
    for i in range(150):
        model.cleargrads()
        loss = model(trainX,trainY)
        train_loss.append(np.float32(loss.data))

        loss.backward()
        optimizer.update()

        loss = model(testX, testY)
        test_loss.append(np.float32(loss.data))



    plt.plot(np.array(train_loss)/len(trainX))
    plt.plot(np.array(test_loss)/len(testX))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train_loss','test_loss'])
    plt.show()


    train_accuracy = np.sum(model.predict(trainX) == np.argmax(trainY,axis=1)) / len(trainX)
    test_accuracy = np.sum(model.predict(testX) == np.argmax(testY,axis=1)) / len(testX)

    print('train_accuracy:',train_accuracy)
    print('test_accuracy:',test_accuracy)





__author__ = 'Daisuke Yoda'
__Date__ = 'January 2019'


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from chainer import Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L

from gensim.models.keyedvectors import KeyedVectors


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
        pad_vec = [26] * pad_len
        sentence.extend(pad_vec)
        paded_vec.append(sentence)

    return np.array(paded_vec, dtype=np.int32)


class LSTM(Chain):
    def __init__(self, in_size, hidden_size,out_size):
        super(LSTM, self).__init__(
            h1 = L.NStepLSTM (
                n_layers=1,
                in_size=in_size,
                out_size=hidden_size,
                dropout=0.5),
            hy = L.Linear(hidden_size*17,out_size))


    def __call__(self,input_data,hx=None):
        hx = hx.reshape(1,-1,self.h1.out_size)
        input_x = [Variable(x) for x in input_data]
        hx,cx,y = self.h1(hx,None,input_x)
        y2 = [F.concat(x, axis=0) for x in F.pad_sequence(y,length=17, padding=0.)]
        y2 = F.concat([F.expand_dims(x,axis=0) for x in y2],axis=0)

        out = self.hy(y2)

        return out

    def predict(self,word,hx=None):
        test_vec = word_to_index(word)
        test_vec = one_hot_encoding(test_vec).astype(np.float32)
        res = self([test_vec],hx=None)[0]
        return F.argmax(res)



if __name__ == '__main__':
    word_vectors = KeyedVectors.load_word2vec_format("trainer/glove.6B.200d.bin")

    df = pd.read_csv ('trainer/split_point_2.csv', index_col=0)
    df = df[np.random.permutation (df.columns)]

    word_vec = np.array ([word_vectors.get_vector (word) for word in df.columns], dtype=np.float32)

    original_data = [word_to_index (x) for x in df.columns]
    original_data = [one_hot_encoding (x).astype (np.float32) for x in original_data]

    split_point = np.nan_to_num (df, 0).T

    trainX = original_data[:1500]
    testX = original_data[1500:]
    trainY = split_point[:1500].astype(np.float32)
    testY = split_point[1500:].astype(np.float32)

    model = LSTM(27, 200, 17)
    serializers.load_npz('model5.npz',model)


    train_accuracy = 100*np.sum(np.argmax(model(trainX,word_vec[:1500]).data, axis=1)==np.argmax(trainY,axis=1))/len(trainX)
    test_accuracy = 100*np.sum(np.argmax(model(testX,word_vec[1500:]).data, axis=1)==np.argmax(testY,axis=1))/len(testX)

    print('train_accuracy:',train_accuracy)
    print('test_accuracy:',test_accuracy)


__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'


import pandas as pd
import numpy as np

from chainer import Chain, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L


def word_to_index(word):
    word_index = [ord (char) - 97 for char in word]
    return word_index


def one_hot_encoding(indices, n_class=27):
    emb_zero = (indices != 26).astype (int)

    return emb_zero.reshape (-1, 1) * np.eye (n_class)[indices]

def padding(sentences):
    max_len = np.max([len(s) for s in sentences])
    paded_vec = []
    for sentence in sentences:
        pad_len = max_len - len(sentence)
        pad_vec = [26] * pad_len
        sentence.extend(pad_vec)
        paded_vec.append(sentence)

    return np.array (paded_vec, dtype=np.int32)


class LSTM(Chain):
    def __init__(self, in_size, hidden_size,hidden2_size,hidden3_size,out_size):
        super(LSTM, self).__init__(
            xh=L.Linear(in_size, hidden_size),
            hh=L.LSTM(hidden_size, hidden2_size),
            hh2=L.LSTM(hidden2_size, hidden3_size),
            hy=L.Linear(hidden3_size,out_size))

    def forward(self,x):
        x = Variable(x)
        x = F.dropout(x,0.3)
        h = self.xh(x)
        h = F.tanh(h)
        h2 = self.hh(h)
        h2 = F.dropout(h2, 0.3)
        h2 = self.hh2(h2)
        h2 = F.dropout(h2, 0.3)
        y = self.hy(h2)

        return y

    def reset(self):
        self.hh.reset_state()


if __name__ == '__main__':
    df = pd.read_csv('trainer/split_point.csv',index_col=0)
    training_data = [word_to_index(x) for x in df.columns]
    training_data = padding(training_data).T
    training_data = [one_hot_encoding (x).astype(np.float32) for x in training_data]

    split_point = np.nan_to_num(df.values,0).astype(np.int32)

    model = LSTM(27,30,50,30,27)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    for i in range(1000):
        model.cleargrads()
        model.reset()
        for X,Y in zip(training_data,split_point):
            pred = model(X)
            loss = F.softmax_cross_entropy(pred,Y)

        loss.backward()
        optimizer.update()


    serializers.save_npz("model_new4.npz", model)



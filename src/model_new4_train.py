__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'


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
        pad_vec = [26] * pad_len
        sentence.extend(pad_vec)
        paded_vec.append(sentence)

    return np.array(paded_vec, dtype=np.int32)


class LSTM(Chain):
    def __init__(self, in_size, hidden_size,out_size):
        super(LSTM, self).__init__(
            h1 = L.NStepLSTM (
                n_layers=2,
                in_size=in_size,
                out_size=hidden_size,
                dropout=0.3),
            hy = L.Linear(hidden_size,out_size))
        self.hx = None
        self.cx = None


    def __call__(self,input_data):
        input_x = [Variable(x) for x in input_data]
        hx,cx,y = self.h1(None,None,input_x)
        y2 = [self.hy(item) for item in y]

        return y2

    def predict(self,word):
        test_vec = word_to_index(word)
        test_vec = one_hot_encoding(test_vec).astype(np.float32)
        res = model([test_vec])[0]
        pred = F.argmax(res, axis=1)
        print(pred)



if __name__ == '__main__':
    df = pd.read_csv('trainer/split_point.csv',index_col=0)
    training_data = [word_to_index(x) for x in df.columns]
    training_data = [one_hot_encoding(x).astype(np.float32) for x in training_data]

    split_point = [df[col].dropna().values for col in df.columns]
    model = LSTM(27,50,2)
    optimizer = optimizers.Adam()
    optimizer.setup(model)


    loss_record = []
    for i in range(500):
        loss = 0
        model.cleargrads()
        res = model(training_data)
        for r,ans in zip(res,split_point):
            loss += F.softmax_cross_entropy(r,ans.astype(np.int32))
        loss.backward()
        loss_record.append(float(loss.data))
        optimizer.update()


    plt.plot(loss_record)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    pred = model.predict('age')
    print(pred)





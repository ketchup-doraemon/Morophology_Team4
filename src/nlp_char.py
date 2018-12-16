"""
Created on Fri Dec  7 15:57:26 2018
@author: dy248
"""

import numpy as np
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L


# The function converts the word to a figure.
def word2fig(word):
    return np.array([ord(char) - 97 for char in word])


class LSTM(Chain):
    def __init__(self, in_size, hidden_size, hidden2_size, out_size):
        super(LSTM, self).__init__(
            xh=L.EmbedID(in_size, hidden_size),
            hh=L.LSTM(hidden_size, hidden2_size),
            hh2=L.Linear(hidden2_size, hidden2_size),
            hy=L.Linear(hidden2_size, out_size),
        )

    def __call__(self, x, t):
        x = Variable(x)
        t = Variable(t)

        h = self.xh(x)
        h = F.dropout(h,0.1)
        h = F.tanh(h)
        h = self.hh(h)
        h = self.hh2(h)
        h = F.dropout(h,0.1)
        y = F.relu(self.hy(h))

        return F.softmax_cross_entropy(y, t)

    def predict(self, x):
        x = Variable(x)

        h = self.xh(x)
        h = F.tanh(h)
        h = self.hh(h)
        h = self.hh2(h)
        h = F.relu(self.hy(h))
        y = F.softmax(h)

        return y.data

    def reset(self):
        #self.cleargrads()
        self.hh.reset_state()


if __name__ == '__main__':

    model = LSTM(27, 50, 10, 27)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    training_data = ['create', 'creative', 'creation', 'created', 'creating', 'creats', 'creater']
    training_data = [np.hstack([word2fig(x), 26]).astype(np.int32) for x in training_data]

    loss_record = []
    model.cleargrads()
    for i in range(800):
        model.reset()

        traning_data = np.random.permutation(training_data)
        loss = 0
        for trainX in training_data:
            model.hh.reset_state()
            for j in range(trainX.shape[0] - 1):
                x = np.array([trainX[j]])
                t = np.array([trainX[j + 1]])
                loss += model(x=x, t=t)

        loss_record.append(int(loss.data))
        loss.backward()
        optimizer.update()


    plt.plot(loss_record)
    plt.show()
    print({chr(trainX[0]+97):1.0})

    char = np.array([trainX[0]])
    model.reset()

    while True:
        y = model.predict(char)

        pred = np.apply_along_axis(chr,0, np.argsort(y) + 97)[::-1]
        pred = np.where(pred=='{','end',pred)
        prob = np.sort(y)[0][::-1]
        prob = np.round(prob,5)
        cadidate = {char:p for char,p in zip(pred[:3],prob[:3])}
        print(cadidate)

        char = np.array([np.argmax(y)], dtype=np.int32)

        if char[0] == 26:
            break





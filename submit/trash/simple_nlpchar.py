__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'


import numpy as np
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
from copy import deepcopy


# The function converts the word to a figure.
def word_to_index(word):
    word_index = [ord (char) - 97 for char in word]

    return word_index

def one_hot_encoding(indices,n_class=27):
    return np.eye(n_class)[indices]



class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(LSTM, self).__init__(
            xh=L.Linear(in_size, hidden_size),
            hh=L.GRU(hidden_size, hidden_size),
            hy = L.Linear(hidden_size,out_size),
        )

    def __call__(self, x, t):
        x = one_hot_encoding(x).astype(np.float32)
        x = Variable(x)
        t = Variable(t)
        h = self.xh(x)
        h = F.dropout(h,0.3)
        h = F.tanh(h)
        h2 = self.hh(h)
        h = F.dropout(h2, 0.3)
        y = self.hy(h2)


        return F.softmax_cross_entropy(y, t,reduce='no')

    def predict(self, x):
        x = one_hot_encoding(x).astype (np.float32)
        x = Variable(x)

        h = self.xh(x)
        h = F.tanh(h)
        h2 = self.hh(h)
        y = self.hy(h2)

        y = F.softmax(y)

        return y.data

    def reset(self):
        #self.cleargrads()
        self.hh.reset_state()

def padding(sentences):
    max_len = np.max([len(s) for s in sentences])
    paded_vec = []
    for sentence in sentences:
        pad_len = max_len - len(sentence)
        pad_vec = [26] * pad_len
        sentence.extend(pad_vec)
        paded_vec.append(sentence)

    return np.array(paded_vec,dtype=np.int32)

if __name__ == '__main__':

    model = LSTM(27,100,27)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    training_data = ['create', 'creative', 'creation', 'created', 'creating', 'creats', 'creater']
    #training_data = ['work', 'worked', 'working']
    training_data = [word_to_index(x) for x in training_data]

    loss_record = []
    model.cleargrads()
    for i in range(100):
        model.reset()
        training_sample = deepcopy(np.random.permutation(training_data))
        training_sample = padding(training_sample).T
        loss = 0
        trainX = training_sample[:-1]
        trainY = training_sample[1:]
        for X,Y in zip(trainX,trainY):
            loss += F.sum(model(X,Y))

        loss_record.append(int(loss.data))
        loss.backward()
        optimizer.update()


    plt.plot(loss_record)
    plt.show()


    print({chr(trainX[0][0]+97):1.0})

    char = np.array([trainX[0][0]])
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


    model.reset()
    word = 'creating'
    word_ix = word_to_index(word)
    cr_prob = []
    for char_ix in word_ix:
        y = model.predict(np.array([char_ix]))
        pred = np.apply_along_axis(chr, 0, np.argsort(y) + 97)[::-1]
        pred = np.where(pred=='{','end',pred)
        prob = np.sort(y)[0][::-1]
        prob = np.round(prob,5)
        cadidate = {char:p for char,p in zip(pred[:3],prob[:3])}
        print(cadidate)

        cr_prob.append(prob[0])

    cr_prob = cr_prob[:-1]
    cr_prob = np.cumsum(cr_prob)/np.arange(1,len(word))
    print(cr_prob)


__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'

import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links as L
from copy import deepcopy

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format("trainer/glove.6B.200d.bin")


class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(LSTM, self).__init__(
            xh=L.Linear(in_size, hidden_size),
            hh=L.GRU(hidden_size, hidden_size),
            hy = L.Linear(hidden_size,out_size))

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
        self.hh.reset_state()


def make_pairs(words, max_len=6):
    patterns = defaultdict (list)

    for word in words:
        for second_word in words:
            if word != second_word:
                i = 1
                while (word[:i] == second_word[:i]):
                    i += 1
                if i != 1 and i > max (len (word[i - 1:]), len (second_word[i - 1:])) < max_len:
                    if ("suffix", word[i - 1:], second_word[i - 1:]) in patterns:
                        patterns[("suffix", word[i - 1:], second_word[i - 1:])].append ((word, second_word))
                    else:
                        patterns[("suffix", word[i - 1:], second_word[i - 1:])] = [(word, second_word)]

                i = 1
                while (word[-i:] == second_word[-i:]):
                    i += 1
                if i != 1 and max (len (word[:-i + 1]), len (second_word[:-i + 1])) < max_len:
                    if ("prefix", word[:-i + 1], second_word[:-i + 1]) in patterns:
                        patterns[("prefix", word[:-i + 1], second_word[:-i + 1])].append ((word, second_word))
                    else:
                        patterns[("prefix", word[:-i + 1], second_word[:-i + 1])] = [(word, second_word)]

    return patterns


def molph_classify(thepairs, model, threshold=0.5, min_category=5):
    new_pairs = defaultdict (list)

    for key in thepairs:
        cadidates = thepairs[key]

        similality = []
        for pair in cadidates:
            try:
                cos_sim = model (pair[0], pair[1])
            except:
                pass
            else:
                if cos_sim > threshold:
                    similality.append (pair + (cos_sim,))

        if len (similality) > min_category:
            new_pairs[key] = similality

    return new_pairs


def word2indices(word):
    return np.array([ord(char) - 97 for char in word])


def padding(sentences):
    max_len = np.max([len(s) for s in sentences])
    paded_vec = []
    for sentence in sentences:
        pad_len = max_len - len(sentence)
        pad_vec = [26] * pad_len
        sentence.extend(pad_vec)
        paded_vec.append(sentence)

    return np.array(paded_vec,dtype=np.int32)



def make_same_group(pairs, word):
    pair_list = sum (list (pairs.values ()), [])
    group = [pair[0:2] for pair in pair_list if word == (pair[0] or pair[1])]

    return group


def get_hit_rate(pairs,words):
    hit_word = []
    pair_list = sum(list(pairs.values()),[])
    for word in words:
        hit_word.extend([pair[1] for pair in pair_list if word == pair[0]])
        hit_word.extend([pair[0] for pair in pair_list if word == pair[1]])

    c = Counter(hit_word)
    return zip(*c.most_common(80))

def clustering_group(pairs):
    pair_list1 = [set(pair[0:2]) for pair in sum(list(pairs.values()),[])]
    pair_list2 = deepcopy(pair_list1)
    cluster_list = []
    for pair1 in pair_list1[:]:
        ald = False
        for cluster in cluster_list:
            if pair1 <= cluster:
                ald = True
        if ald:
            continue
            
        while True:       
            calc = False
            for pair2 in pair_list2[:]:
                if pair1 >= pair2:
                    pass
                elif not pair1.isdisjoint(pair2):
                    pair1 = pair1 | pair2
                    calc = True
            if not calc:
                cluster_list.append(pair1)
                break
        
    return cluster_list
        
                     
"""                    
def clustering_group_n(pairs,words):
    pair_list = [pair[0:2] for pair in sum(list(pairs.values()),[])]
    words_list = list(words)
    cluster_list = []
    for word in words_list[:]:
        if word == 'appeared':
            print(word)
        cluster = [word]
        for i in range(len(pair_list)):
            if word == pair_list[i][0]:
                cluster.append(pair_list[i][1])
                pair_list.remove(pair_list[i])
                if pair_list[i][1] in words_list:
                    words_list.remove(pair_list[i][1])

                    
            elif word == pair_list[i][1]:
                cluster.append(pair_list[i][0])
                pair_list.remove(pair_list[i])
                if pair_list[i][0] in words_list:
                    words_list.remove(pair_list[i][0])
                    
        
        if len(cluster) > 1:
            cluster_list.append(cluster)
        
        
    return np.array([np.unique(pair) for pair in np.unique(cluster_list)])
"""

def one_hot_encoding(indices,n_class=27):
    return np.eye(n_class)[indices]


def word_to_index(word):
    word_index = [ord (char) - 97 for char in word]

    return word_index


def occurrence_probability(word_set,pd_frame=True):
    training_data = [word_to_index(x) for x in word_set]

    model = LSTM(27,100,27)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    for i in range(200):
        model.cleargrads()
        model.reset()
        training_sample = deepcopy(np.random.permutation(training_data))
        training_sample = padding(training_sample).T
        loss = 0
        trainX = training_sample[:-1]
        trainY = training_sample[1:]
        for X,Y in zip(trainX,trainY):
            loss += F.sum(model(X,Y))

        loss.backward()
        optimizer.update()
        
    occurrence_probability = {}
    for word in word_set:    
        model.reset()
        word_ix = word_to_index(word)
        word_ix.extend([26])
        char_X = word_ix[:-1]
        char_Y = word_ix[1:]
        cr_prob = []
        for char_x,char_y in zip(char_X,char_Y):
            y = model.predict(np.array([char_x]))
            prob = y[0][char_y]
            cr_prob.append(prob)
            
        cr_prob = np.cumsum(cr_prob)/np.arange(1,len(word)+1)
        
        if pd_frame:
            occurrence_probability[word] = pd.Series(cr_prob)
        else:
            occurrence_probability[word] = cr_prob
            
            
    if pd_frame:
        return pd.DataFrame(occurrence_probability)
    else:
        return occurrence_probability
    
    


if __name__ == '__main__':
    with open ('english_brown.txt') as f:
        data = f.read ()
        data = data.replace('.', '')
        data = data.replace(',', '')
        data = data.replace('""', '')
        data = data.lower()

    all_words = data.split()
    words_set = np.unique(all_words)

    words_set = np.array([word for word in words_set if word.isalpha()])

    model = word_vectors.similarity
    original_pair = make_pairs(words_set, max_len=6)
    pairs = molph_classify(original_pair,model,threshold=0.7,min_category=5)
    """
    word_dic = np.unique(np.ravel([pair[0:2] for pair in sum(list(pairs.values()),[])]))
    common_words,counts = get_hit_rate(pairs,word_dic)
    train_set = [set(np.ravel(make_same_group(pairs,word))) for word in common_words]
    dic_ix = np.array([word2indices(word) for word in word_dic])
    """
    
    morph_cluster = clustering_group(pairs)
    morph_cluster = [pair for pair in morph_cluster if len(pair)>2]
    

    prob = pd.concat([occurrence_probability(pair) for pair in morph_cluster],axis=1)
    prob.to_csv('trainer/occurrence_probability.csv')


    
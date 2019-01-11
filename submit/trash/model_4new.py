__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'

import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from itertools import combinations
from copy import deepcopy

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format("trainer/glove.6B.200d.bin")


def make_pairs(words, max_len=6):
    patterns = defaultdict (list)

    for word in words:
        for second_word in words:
            if word != second_word:
                i = 1
                while word[:i] == second_word[:i]:
                    i += 1
                if i != 1 and i > max(len(word[i - 1:]), len(second_word[i - 1:])) < max_len:
                    if ("suffix", word[i - 1:], second_word[i - 1:]) in patterns:
                        patterns[("suffix", word[i - 1:], second_word[i - 1:])].append ((word, second_word))
                    else:
                        patterns[("suffix", word[i - 1:], second_word[i - 1:])] = [(word, second_word)]

                i = 1
                while word[-i:] == second_word[-i:]:
                    i += 1
                if i != 1 and max(len(word[:-i + 1]), len(second_word[:-i + 1])) < max_len:
                    if ("prefix", word[:-i + 1], second_word[:-i + 1]) in patterns:
                        patterns[("prefix", word[:-i + 1], second_word[:-i + 1])].append ((word, second_word))
                    else:
                        patterns[("prefix", word[:-i + 1], second_word[:-i + 1])] = [(word, second_word)]

    return patterns


def molph_classify(thepairs, model, threshold=0.5, min_category=5):
    new_pairs = defaultdict(list)

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
    return np.array ([ord (char) - 97 for char in word])


def padding(sentences):
    max_len = np.max([len(s) for s in sentences])
    paded_vec = []
    for sentence in sentences:
        pad_len = max_len - len(sentence)
        pad_vec = [26] * pad_len
        sentence.extend(pad_vec)
        paded_vec.append(sentence)

    return np.array (paded_vec, dtype=np.int32)


def make_same_group(pairs, word):
    pair_list = sum (list(pairs.values ()), [])
    group = [pair[0:2] for pair in pair_list if word == (pair[0] or pair[1])]

    return group


def get_hit_rate(pairs, words):
    hit_word = []
    pair_list = sum(list(pairs.values ()), [])
    for word in words:
        hit_word.extend([pair[1] for pair in pair_list if word == pair[0]])
        hit_word.extend([pair[0] for pair in pair_list if word == pair[1]])

    c = Counter (hit_word)
    return zip (*c.most_common (80))


def clustering_group(pairs):
    pair_list1 = [set(pair[0:2]) for pair in sum(list(pairs.values()), [])]
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
                elif not pair1.isdisjoint (pair2):
                    pair1 = pair1 | pair2
                    calc = True
            if not calc:
                cluster_list.append (pair1)
                break

    return cluster_list

def get_stem(cluster):
    stem_candidate = []
    for pair in combinations(cluster,2):
        stem_candidate.append([char1 for char1,char2 in zip(pair[0],pair[1]) if char1 == char2])

    stem = sorted(stem_candidate,key=lambda x: len(x))

    return ''.join(stem[0])


def vector_by_stem(stem_cluster,pd_frame=True):
    split_vec = {}
    for stem in stem_cluster.keys():
        for word in stem_cluster[stem]:
            if word[:len(stem)] == stem:
                vec = np.eye(len(word))[len(stem) - 1]


            elif word[::-1][:len(stem[::-1])] == stem[::-1]:
                vec = np.eye(len(word))[len(stem)][::-1]


            if pd_frame:
                split_vec[word] = pd.Series(vec)
            else:
                split_vec[word] = vec

    return pd.DataFrame(split_vec)






if __name__ == '__main__':
    with open('english_brown.txt') as f:
        data = f.read()
        data = data.replace('.', '')
        data = data.replace(',', '')
        data = data.replace('""', '')
        data = data.lower()

    all_words = data.split()
    words_set = np.unique(all_words)

    words_set = np.array([word for word in words_set if word.isalpha()])

    model = word_vectors.similarity
    original_pair = make_pairs(words_set, max_len=6)
    pairs = molph_classify(original_pair, model, threshold=0.7, min_category=5)

    morph_cluster = clustering_group(pairs)
    morph_cluster = [pair for pair in morph_cluster if len(pair) > 1]
    morph_cluster = [sorted(pair,key=lambda x: len(x)) for pair in morph_cluster]

    stem_cluster = {get_stem(cluster):cluster for cluster in morph_cluster}
    split_list = vector_by_stem(stem_cluster)

    split_list.to_csv('trainer/point_list_2.csv')





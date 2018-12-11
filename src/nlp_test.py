# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:27:55 2018

@author: dy248
"""

from collections import defaultdict
import numpy as np
from gensim.models.keyedvectors import KeyedVectors 
from gensim.scripts.glove2word2vec import glove2word2vec

glove2word2vec("trainer\glove.6B.200d.txt","trainer\glove.6B.200d.bin")
word_vectors = KeyedVectors.load_word2vec_format("trainer\glove.6B.200d.bin")


from matplotlib import pyplot as plt
import networkx as nx

def make_pairs(words, max_len =6):
    patterns = defaultdict(list)

    for word in words:
        for second_word in words:
            if word != second_word:
                i = 1
                while(word[:i]==second_word[:i]):
                    i += 1
                if i != 1 and i > max(len(word[i-1:]), len(second_word[i-1:])) < max_len:
                    if ("suffix", word[i-1:], second_word[i-1:]) in patterns:
                        patterns[("suffix", word[i-1:], second_word[i-1:])].append((word, second_word))
                    else:
                        patterns[("suffix", word[i-1:], second_word[i-1:])] = [(word, second_word)]
                        
                i = 1
                while(word[-i:]==second_word[-i:]):
                    i += 1
                if i != 1 and max(len(word[:-i+1]), len(second_word[:-i+1])) < max_len:
                    if ("prefix", word[:-i+1], second_word[:-i+1]) in patterns:
                        patterns[("prefix", word[:-i+1], second_word[:-i+1])].append((word, second_word))
                    else:
                        patterns[("prefix", word[:-i+1], second_word[:-i+1])] = [(word, second_word)]
                        
    return patterns

def molph_classify(pairs,threshold=0.5,min_category=5):
    new_pairs = defaultdict(list)

    for key in pairs:
        cadidates = pairs[key]
        
        similality = []
        for pair in cadidates:
            try:
                cos_sim = word_vectors.similarity(pair[0],pair[1])
            except:
                pass
            else: 
                if cos_sim > threshold:
                    similality.append(pair + (cos_sim,))
        
        if len(similality) > min_category :
             new_pairs[key] = similality
             
    return new_pairs

def make_same_group(pairs,word):
    pair_list = sum(list(pairs.values()),[])
    group = []
    for pair in pair_list:
        if word == (pair[0] or pair[1]):
            group.append(pair)
            
    return group
    


if __name__ == '__main__':
    
    with open('english_brown.txt') as f:
        data = f.read()
        data = data.replace('.','')
        data = data.replace(',','')
        data = data.replace('""','')
        data = data.lower()
        
    all_words = data.split()
    words_set = np.unique(all_words)[726:]
    

    model = word_vectors.similarity
    original_pair = make_pairs(words_set, max_len =6)
    pairs = molph_classify(original_pair,model)
    group = make_same_group(pairs,'work')


    G = nx.Graph()
    for pair in group:  
        G.add_nodes_from([pair[0],pair[1]])
        G.add_edge(pair[0],pair[1])
    
    
    plt.figure(figsize=(7,7))
    pos = nx.spring_layout(G,k=0.7)
    nx.draw_networkx_nodes(G, pos, node_color="0.9", node_size=5000.0)
    nx.draw_networkx_edges(G, pos, edge_color="0.1", width=2.5)
    nx.draw_networkx_labels(G, pos, fontsize=25, font_weight="bold")
    plt.axis('on')
    plt.show()
    
    
    
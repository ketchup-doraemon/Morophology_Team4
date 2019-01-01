__author__ = 'Daisuke Yoda'
__Date__ = 'December 2018'

from collections import defaultdict
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format(r"trainer/glove.6B.100d.bin")



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


def molph_classify(thepairs,model,threshold=0.5,min_category=5):
    new_pairs = defaultdict(list)

    for key in thepairs:
        cadidates = thepairs[key]
        
        similality = []
        for pair in cadidates:
            try:
                cos_sim = model(pair[0],pair[1])
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
    group = [pair for pair in pair_list if word==(pair[0] or pair[1])]
            
    return group


def plot_graph(pair_group):
    G = nx.Graph()
    for pair in pair_group:
        G.add_nodes_from([pair[0],pair[1]])
        G.add_edge(pair[0],pair[1])
    
    plt.figure(figsize=(7,7))
    pos = nx.spring_layout(G,k=0.7)
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=5000.0)
    nx.draw_networkx_edges(G, pos, width=2.5)
    nx.draw_networkx_labels(G, pos, fontsize=25, font_weight="bold")
    plt.axis('on')
    plt.show()
       


if __name__ == '__main__':
    
    with open('english_brown.txt') as f:
        data = f.read()
        data = data.replace('.','')
        data = data.replace(',','')
        data = data.replace('""','')
        data = data.lower()
        
    all_words = data.split()
    words_set = np.unique(all_words)
    words_set = [word for word in words_set if word.isalpha()]

    model = word_vectors.similarity
    original_pair = make_pairs(words_set, max_len =6)
    pairs = molph_classify(original_pair,model,threshold=0.7,min_category=5)
    group = make_same_group(pairs,'work')
    plot_graph(group)



# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 23:17:00 2018

@author: dy248
"""

import pandas as pd

df = pd.read_csv('trainer/occurrence_probability.csv',index_col=0)


def reward(word):
    return df[word]*[1.01**x for x in range(df.shape[0])]
    
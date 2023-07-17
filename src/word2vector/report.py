import pandas as pd
import csv
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import ast
import numpy as np
import nltk
import pickle
from src.word2vector.treebank import *

nltk.download('punkt')      


def save_data(data_arr,columns_arr,directory):
  df = pd.DataFrame(data_arr, columns=columns_arr)
  df.to_csv(directory, index=False, quoting=csv.QUOTE_ALL)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def load_saved_params(category):
    """
    A helper function that loads previously saved parameters and resets
    iteration start.
    """
    params_file = f"models/{category}.word2vec.npy"
    params = np.load(params_file)
    return params

if __name__ == '__main__':

    category_dict = {'action':set(),'adventure':set(),'rpg':set(),'strategy':set(),'simulation':set(),'sports_and_racing':set()}
    categories = ['action','adventure','rpg','strategy','simulation','sports_and_racing']

    for category in categories:
        df = pd.read_csv(f'data/traindata/{category}.csv')
        data = df.to_dict('dict')
        for about in data['about'].values():
            for sentence in nltk.sent_tokenize(about):
                splitted = sentence.strip().split()[1:]
                # Deal with some peculiar encoding issues with this file
                words = [w.lower() for w in splitted]
                for word in words:
                    category_dict[category].add(word)


    common_words = set()

    for word in category_dict['action']:
      count = 0
      for category in categories:
        if word in category_dict[category]:
          count+=1
      if(count==6):
        common_words.add(word)    

    sim_result = []
    for word in common_words:
      for i in range(6):
        for j in range(i+1,6):
          cat1 , cat2 = categories[i],categories[j]
          params1,params2 = load_saved_params(cat1),load_saved_params(cat2)
          dataset1,dataset2 = SteamData(f'data/traindata/{cat1}.csv'), SteamData(f'data/traindata/{cat2}.csv')
          idx1 , idx2 = dataset1.tokens()[word], dataset2.tokens()[word]
          res =cosine_similarity(params1[idx1],params2[idx2])
          sim_result.append([word,cat1,cat2,res])

    save_data(sim_result,['word','category1','category2','cosine similarity'],f'stats/cosine_similarity.csv')
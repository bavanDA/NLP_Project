import pandas as pd
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import ast
import numpy as np


print("start: ")

categories = ['action','adventure','rpg','strategy','simulation','sports_and_racing']
directory = 'stats/'

#count labels 
labels = 6


# count sentences
sentences_count = 0 
result1 = defaultdict(int)
result2 = defaultdict(int)

print("sentences count")

for category in categories:
  df = pd.read_csv(f'data/sentencebroken/{category}/{category}.csv')
  data = df.to_dict('dict')
  for i in range(len(data['sentences'])):
    result1[category]+=1
    result2[category]+=len(ast.literal_eval(data['sentences'][i]))

  print(f'    {category}: {result2[category]}')
  sentences_count+=result2[category]

print("    ---------------------")
print("    total: " , sentences_count)

df1 = pd.DataFrame.from_dict([result1])
df1.to_csv(f'{directory}data_count.csv', index=False, quoting=csv.QUOTE_ALL)

df2 = pd.DataFrame.from_dict([result2])
df2.to_csv(f'{directory}sentences_count.csv', index=False, quoting=csv.QUOTE_ALL)



#count words 
words_count = 0 
print("words count")
result3 = defaultdict(int)

category_dict = {'action':set(),'adventure':set(),'rpg':set(),'strategy':set(),'simulation':set(),'sports_and_racing':set()}
words_dict = defaultdict(int)
for category in categories:
  df = pd.read_csv(f'data/wordbroken/{category}/{category}.csv')
  data = df.to_dict('dict')
  for i in range(len(data['words'])):
    words_arr = ast.literal_eval(data['words'][i])
    result3[category]+= len(words_arr)
    for word in words_arr: 
      words_dict[word]+=1
      category_dict[category].add(word)

  print(f'    {category}: {result3[category]}')
  words_count+=result3[category]

  
words_dict = dict(sorted(words_dict.items(), key=lambda x: x[1],reverse=1))

print("    ---------------------")
print("    total: " , words_count)

df3 = pd.DataFrame.from_dict([result3])
df3.to_csv(f'{directory}words_count.csv', index=False, quoting=csv.QUOTE_ALL)



#count unique words 
print("total unique words ", len(words_dict.keys()))


common_words = 0
uncommon_words = 0 
common_dict = set()

for word in words_dict.keys():
  count = 0
  for category in categories:
    if word in category_dict[category]:
      count+=1
  if(count>1):
    common_words+=1
    common_dict.add(word)
  else:
    uncommon_words+=1


# count common and uncommon words
print(f"total common words count: {common_words}")
print(f"total uncommon words count: {uncommon_words}")



# for each category:
result4 = []

for category in categories:
  common_words = 0
  uncommon_words = 0 
  for word in category_dict[category]:
    if word in common_dict:
      common_words+=1
    else:
      uncommon_words+=1
  result4.append([category,len(category_dict[category]),common_words,uncommon_words])
  print(f"{category}: ")
  print(f"      unique words count: {len(category_dict[category])}")
  print(f"      common words count: {common_words}")
  print(f"      uncommon words count: {uncommon_words}")

  
df4 = pd.DataFrame(result4, columns=['category', 'unique words','common words','uncommon'])
df4.to_csv(f'{directory}unique_common_uncommon_count.csv', index=False, quoting=csv.QUOTE_ALL)



columns=['category']
columns.extend([f'word{i}'for i in range(1,11)])


# 10 uncommon words
print("10 uncommon words:")
dfa=defaultdict(int)
result5 = []
for category in categories:
  df = pd.read_csv(f'data/wordbroken/{category}/{category}.csv')
  data = df.to_dict('dict')
  tmp_dict = defaultdict(int)
  for i in range(len(data['words'])):
    words_arr = ast.literal_eval(data['words'][i])
    for word in words_arr: 
      if word not in common_dict:
        tmp_dict[word]+=1
  tmp_dicts = dict(sorted(tmp_dict.items(), key=lambda x: x[1],reverse=1)[:10])
  tmp = [category]
  tmp.extend(tmp_dicts.keys())
  result5.append(tmp)
  print(category+":")
  print("  ",tmp_dicts)
  
df5 = pd.DataFrame(result5, columns=columns)
df5.to_csv(f'{directory}ten_uncommon_words.csv', index=False, quoting=csv.QUOTE_ALL)


result6 = []
print("10 common words:")
words_dict = dict(words_dict)
for category in categories:
  df = pd.read_csv(f'data/wordbroken/{category}/{category}.csv')
  data = df.to_dict('dict')
  tmp_dict = defaultdict(int)
  for i in range(len(data['words'])):
    words_arr = ast.literal_eval(data['words'][i])
    for word in words_arr: 
      if word in common_dict:
        tmp_dict[word]+=1

  for word in tmp_dict.keys():
    count_word = words_dict[word]
    count_word_label = tmp_dict[word]
    tmp_dict[word] = (count_word/count_word_label)/(count_word/(count_word-count_word_label))
  tmp_dicts = dict(sorted(tmp_dict.items(), key=lambda x: x[1],reverse=1)[:10])
  tmp = [category]
  tmp.extend(tmp_dicts.keys())
  result6.append(tmp)
  print(category+":")
  print("  ",tmp_dicts)

df6 = pd.DataFrame(result6, columns=columns)
df6.to_csv(f'{directory}ten_common_words.csv', index=False, quoting=csv.QUOTE_ALL)



result7 = []
for category in categories:
  df = pd.read_csv(f'data/wordbroken/{category}/{category}.csv')

  data = df.to_dict('dict')
  df_dict = defaultdict(int)
  for i in range(len(data['words'])):
    words_arr = ast.literal_eval(data['words'][i])
    for word in words_arr: 
      df_dict[word]+=1

  tf_idf = defaultdict(int)
  for i in range(len(data['words'])):
    words_arr = ast.literal_eval(data['words'][i])
    for idx, word in enumerate(words_arr): 
        tf = float(words_arr.count(word)/len(words_arr))
        idf = np.log(float(words_arr.count(word))/df_dict[word])
        tmp_dict[word]=tf*idf
  tf_idf = sorted(tf_idf.items(), key=lambda x: x[1],reverse=1)
  result7.append([category,tf_idf[:10]])

df7 = pd.DataFrame(result7, columns=['category', 'words'])
df7.to_csv(f'{directory}tf_idf_words.csv', index=False, quoting=csv.QUOTE_ALL)


hist_show = dict(words_dict[:15])
plt.bar(list(hist_show.keys()), hist_show.values(), color='g')
plt.savefig(f'{directory}hist.png')




import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import csv
import os


print("start: ")

categories = ['action','adventure','rpg','strategy','simulation','sports_and_racing']
nltk.download('punkt')

for category in categories:
  df = pd.read_csv(f'data/clean/{category}/{category}.csv')
  data = df.to_dict('dict')
  sentences_data = []
  words_data = []

  for i in range(len(data['clean_description'])):
      id = data['id'][i]
      description = data['clean_description'][i]
      sentences = nltk.sent_tokenize(description)
      words = word_tokenize(description)
      sentences_data.append([id, sentences])
      words_data.append([id, words])


  directory1 = f'data/sentencebroken/{category}/'
  if not os.path.exists(directory1):
    os.makedirs(directory1)

  directory2 = f'data/wordbroken/{category}/'
  if not os.path.exists(directory2):
    os.makedirs(directory2)

  df1 = pd.DataFrame(sentences_data, columns=['id', 'sentences'])
  df1.to_csv(f'{directory1}{category}.csv', index=False, quoting=csv.QUOTE_ALL)

  df2 = pd.DataFrame(words_data, columns=['id', 'words'])
  df2.to_csv(f'{directory2}{category}.csv', index=False, quoting=csv.QUOTE_ALL)

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import csv

print("start: ")
categories = ['action','adventure','rpg','strategy','simulation','sports_and_racing']
english_letters_regex = re.compile(r'[a-zA-Z]+')

data_count=0
clean_data_count=0
for category in categories:
  df = pd.read_csv(f'data/raw/{category}/{category}.csv')
  data = df.to_dict('dict')
  clean_data = []

  for i in range(len(data['about'])):
      id = data['id'][i]
      description = data['about'][i]
      clean_description = description.replace('\nAbout This Game\n', '')
      data_count+=1
      if(len(clean_description)>10):
        clean_data_count +=1
        clean_data.append([id, clean_description])

  directory = f'data/clean/{category}/'
  if not os.path.exists(directory):
    os.makedirs(directory)

  df1 = pd.DataFrame(clean_data, columns=['id', 'clean_description'])
  df1.to_csv(f'{directory}{category}.csv', index=False, quoting=csv.QUOTE_ALL)


print("before cleaning data :",data_count)
print("after cleaning data:",clean_data_count)




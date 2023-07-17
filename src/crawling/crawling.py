import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import csv


print("start: ")


def add_information(app_ids, category, data):
    for i, app_id in enumerate(app_ids):
        response = requests.get(f"https://store.steampowered.com/app/{app_id}")
        soup = BeautifulSoup(response.content, 'lxml')
        response.close()
        try:
            name = soup.title.string
            info = soup.find("div", {"id": "aboutThisGame"}).get_text("\n")
            info = re.sub('(\r\n|\r|\n)+', '\n', info).replace("\t","")
            data.append([app_id, name, category, info])
        except:
            print(f"An exception occurred app_id = {app_id}")
            print(f"https://store.steampowered.com/app/{app_id}")



categories = ['action','adventure','rpg','strategy','simulation','sports_and_racing']
data = []
for category in categories:
    print(category)
    start = 5000
    while True:
        response = requests.get(f"https://store.steampowered.com/saleaction/ajaxgetsaledynamicappquery?cc=US&l=english&flavor=contenthub_all&start={start}&count=100&tabuniqueid=6&strContentHubType=category&strContentHubCategory={category}")
        app_ids = response.json()["appids"]
        response.close()
        start += 100
        print(start)
        if len(app_ids) == 0:
            break
        add_information(app_ids, category, data)
        if start % 5000 == 0:
            df = pd.DataFrame(data, columns=['id', 'name', 'category', 'about'])
            filename = f'gdrive/MyDrive/NLP/Dataset/{category}{start//5000}.csv'
            df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
            data = []
    if start % 5000 != 0:
        df = pd.DataFrame(data, columns=['id', 'name', 'category', 'about'])
        filename = f'gdrive/MyDrive/NLP/Dataset/{category}{(start//5000)+1}.csv'
        df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL)
        data = []


categories = ['action','adventure','rpg','strategy','simulation','sports_and_racing']
for category in categories:
  PATH = 'Dataset/'

  if not os.path.exists(PATH):
    os.makedirs(PATH)
  
  df1 = pd.read_csv(f'{PATH}{category}1.csv')
  df2 = pd.read_csv(f'{PATH}{category}2.csv')
  df3 = pd.read_csv(f'{PATH}{category}3.csv')

  dict1 = df1.to_dict('dict')
  dict2 = df2.to_dict('dict')
  dict3 = df3.to_dict('dict')

  combined_dict = []
  for i in range(len(dict1['id'])):
      combined_dict.append([dict1['id'][i],dict1['name'][i],dict1['category'][i],dict1['about'][i]])
  for i in range(len(dict2['id'])):
      combined_dict.append([dict2['id'][i],dict2['name'][i],dict2['category'][i],dict2['about'][i]])
  for i in range(len(dict3['id'])):
      combined_dict.append([dict3['id'][i],dict3['name'][i],dict3['category'][i],dict3['about'][i]])
      
  directory = f'data/raw/{category}/'
  df = pd.DataFrame(combined_dict, columns=['id', 'name', 'category', 'about'])
  if not os.path.exists(directory):
    os.makedirs(directory)
  df.to_csv(f'{directory}{category}.csv', index=False, quoting=csv.QUOTE_ALL)

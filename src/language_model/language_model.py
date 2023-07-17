import pandas as pd
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import ast
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
import datetime



class Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
    


def log(message, task):
    if task is None:
        return
    if (not os.path.exists(task)):
        os.makedirs(f'logs', exist_ok=True)
        with open(f'logs/{task}', 'w') as f:
            f.write(f"[{datetime.datetime.now()}] {message.encode('utf-8')}\n")
    else:
        with open(f'logs/{task}', 'a') as f:
            f.write(f"[{datetime.datetime.now()}] {message.encode('utf-8')} \n")


def get_model():
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',eos_token='<|endoftext|>', pad_token='<|pad|>')
  model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
  model.resize_token_embeddings(len(tokenizer))
  return model,tokenizer

def generate_description(tokenizer,model,category):
  generated = tokenizer("<|startoftext|> ", return_tensors="pt").input_ids.cuda()
  sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                  max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
  for i, sample_output in enumerate(sample_outputs):
      print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
      log("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)),f"language_model: {category}")
      # pd.options.display.max_colwidth = 1000
      # descriptions.sample(10)


def get_dataset(category,tokenizer):

  df = pd.read_csv(f'data/traindata/{category}.csv')
  data = df.to_dict('dict')

  descriptions = list(data['about'].values())
  max_length = max([len(tokenizer.encode(description)) for description in descriptions])

  dataset = Dataset(descriptions, tokenizer, max_length=max_length)
  train_size = int(0.9 * len(dataset))
  train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

  return train_dataset, val_dataset

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    training_args = TrainingArguments(output_dir=f'models/', num_train_epochs=1, logging_steps=100, save_steps=5000,
                                    per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                    warmup_steps=10, weight_decay=0.05, logging_dir=f'logs', report_to = 'none')

    categories = ['action','adventure','rpg','strategy','simulation','sports_and_racing']
    for category  in categories:
        print(category)
        model,tokenizer = get_model()
        train_dataset, val_dataset = get_dataset(category,tokenizer)
        torch.cuda.empty_cache()
        Trainer(model=model,  args=training_args, train_dataset=train_dataset,
          eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                  'attention_mask': torch.stack([f[1] for f in data]),
                                                                  'labels': torch.stack([f[0] for f in data])}).train()
        generate_description(tokenizer,model,category)
        torch.save(model.state_dict(), f'models/{category}.language_model.pt')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback, EarlyStoppingCallback
from torch.utils.data import Dataset, random_split
import torch
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import nltk


def load_saved_params(category):
    params_file = f"models/{category}.word2vec.npy"
    params = np.load(params_file)
    return params

class SteamDataset(Dataset):
    def __init__(self, features,labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature,label

def get_dataset():

  categories = ['action','adventure','rpg','strategy','simulation','sports_and_racing']
  labels = []
  word2vecs = []

  for cat_idx,category in enumerate(categories):
    df = pd.read_csv(f'data/traindata/{category}.csv')
    data = df.to_dict('dict')
    idx = 0
    params = load_saved_params(category)
    for about in data['about'].values():
        for sentence in nltk.sent_tokenize(about):
                  splitted = sentence.strip().split()[1:]
                  for w in splitted :
                      labels.append(cat_idx)
                      scale = 100
                      tmp = np.round(params[idx] * scale).astype(int)
                      # tmp = torch.quantize_per_tensor(torch.tensor(params[idx]), 0, 100, dtype=torch.quint8)
                      word2vecs.append(tmp)


  dataset = SteamDataset(word2vecs,labels)
  train_size = int(0.9 * len(dataset))
  train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

  return train_dataset, val_dataset

def my_data_collator(batch):
  """Collate samples from MyDataset into batched Tensors"""

  features = [b[0] for b in batch]
  labels = [b[1] for b in batch]

  features = torch.tensor(features)

  labels = torch.tensor(labels)

  batch = {"input_ids": features, "labels": labels}

  return batch


# You can optionally use this function with the right parameter to compute results.
# You don't have to use this though. You can do the caculation anyway you like.
def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return {"accuracy": np.mean(predictions == labels)}

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")

if __name__ == '__main__':


    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=6)
    train_dataset, val_dataset = get_dataset()

    arguments = TrainingArguments(
        output_dir="sample_hf_trainer",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        evaluation_strategy="epoch", # run validation at the end of each epoch
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        seed=224
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator = my_data_collator,
        tokenizer = tokenizer
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
    trainer.add_callback(LoggingCallback("sample_hf_trainer/log.jsonl"))

    trainer.train()
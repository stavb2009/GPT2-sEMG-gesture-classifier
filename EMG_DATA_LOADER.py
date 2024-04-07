import pandas as pd
import torch
import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import EMG_utils  # Ensure this module has all the required functions
from EMG_ENV import ENV


class CustomDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for EMG gesture recognition using encoded text sequences."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    """Collate function to pad input sequences and attention masks for batch processing."""
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class Data_loader():
    # Data preparation
    data_location = ENV.DATA_LOCATOIN

    data = pd.read_parquet(data_location)

    # sample the data for debugging
    ratio = ENV.SAMPLING_RATIO  # default - all the data
    frac = 1 / ratio
    random_n = random.randint(0, ratio - 1)
    data = data.iloc[round(data.shape[0] * frac) * random_n:round(data.shape[0] * frac) * (random_n + 1)]

    # Simplify data for initial run
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

    # Ensure balanced class representation
    legal_values = ENV.LEGAL_VALUES  # Assuming 8 classes
    value_counts = data['TRAJ_GT'].value_counts()
    min_count = value_counts.min()
    # Sample the dataset to include the minimum count of items from each category
    balanced_data = pd.concat([data[data['TRAJ_GT'] == value].sample(min_count, replace=(
                len(data[data['TRAJ_GT'] == value]) > 0)) if value in data['TRAJ_GT'].unique() else pd.DataFrame(
        columns=data.columns) for value in legal_values])

    # Shuffle the resulting dataset
    balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
    data = balanced_data
    # Split the dataset
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, Valid_data = train_test_split(train_data, test_size=0.2, random_state=42)


    # Tokenization
    tokenizer = GPT2Tokenizer.from_pretrained(ENV.MODEL_NAME)
    tokenizer.padding_side = ENV.PADDING_SIDE
    tokenizer.pad_token = tokenizer.eos_token  # Ensure tokenizer.pad_token_id is not None

    # Prepare encodings
    train_encodings = tokenizer(train_data['input_sequence'].tolist(), padding=True, truncation=True, return_tensors="pt")
    test_encodings = tokenizer(test_data['input_sequence'].tolist(), padding=True, truncation=True, return_tensors="pt")
    Valid_encodings = tokenizer(Valid_data['input_sequence'].tolist(), padding=True, truncation=True, return_tensors="pt")


    # Dataloader preparation
    batch_size = ENV.BATCH_SIZE
    train_dataset = CustomDataset(train_encodings, train_data['TRAJ_GT'].tolist())
    test_dataset = CustomDataset(test_encodings, test_data['TRAJ_GT'].tolist())
    Valid_dataset = CustomDataset(Valid_encodings, test_data['TRAJ_GT'].tolist())


    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    Valid_loader = DataLoader(Valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

DATA_LOADER = Data_loader()

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# Load training data
def load_training_data():
    print("Loading training data...")
    data_dic_path = "dataset"
    x_train = pd.read_csv(f'{data_dic_path}/x_train_time_aware.csv').astype(float)
    y_train = pd.read_csv(f'{data_dic_path}/y_train_time_aware.csv').astype(float)
    print('Training Data Loaded!')
    print('Total Length: ', len(x_train))
    x_train = torch.from_numpy(x_train.values).unsqueeze(1)
    y_train = torch.from_numpy(y_train.values)
    full_dataset = TensorDataset(x_train, y_train)
    return full_dataset

# Select samples to label(manually)
def rs_acquire_label(full_dataset, rs_rate = 0.05):
    num_samples = int(len(full_dataset) * rs_rate)
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    label_set = Subset(full_dataset, indices)
    all_indices = set(range(len(full_dataset)))
    chosen_indices = set(indices.tolist())
    not_chosen_indices = all_indices - chosen_indices
    unlabel_set = Subset(full_dataset, list(not_chosen_indices))
    print("label set size: ", len(label_set))
    print("unlabel set size: ", len(unlabel_set))
    return label_set, unlabel_set

# Split the dataset into training and validation set
def train_val_split(label_set, batch_size):
    n_val = int(len(label_set) * 0.1)
    train_set, val_set = torch.utils.data.random_split(label_set, [len(label_set) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

# Main
def process(batch_size, rs_rate):
    full_dataset = load_training_data()
    label_set, unlabeled_set = rs_acquire_label(full_dataset, rs_rate)
    train_loader, val_loader = train_val_split(label_set, batch_size)
    return train_loader, val_loader, unlabeled_set
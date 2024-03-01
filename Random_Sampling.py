import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, sampler, Subset
from torchvision import datasets, transforms


def load_training_data():
    # Load data
    print("Loading training data...")
    data_dic_path = "dataset"
    x_train = pd.read_csv(f'{data_dic_path}/x_train_time_aware.csv').astype(float)
    y_train = pd.read_csv(f'{data_dic_path}/y_train_time_aware.csv').astype(float)
    print('Training Data Loaded!')
    print('Total Length: ', len(x_train))
    x_train = torch.from_numpy(x_train.values).unsqueeze(1)
    y_train = torch.from_numpy(y_train.values)
    # print("Dataframe to tensor done!")
    full_dataset = TensorDataset(x_train, y_train)
    return full_dataset

def rs_labeler(full_dataset, rs_rate= 0.05):
    num_samples = int(len(full_dataset) * rs_rate)
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    label_set = Subset(full_dataset, indices)

    all_indices = set(range(len(full_dataset)))
    chosen_indices = set(indices.tolist())
    not_chosen_indices = all_indices - chosen_indices
    unlabel_set = Subset(full_dataset, list(not_chosen_indices))
    # print("label set size: ", len(label_set))
    # print("unlabel set size: ", len(unlabel_set))
    return label_set, unlabel_set


def data_loader(label_set):
    n_val = int(len(label_set) * 0.1)
    train_set, val_set = torch.utils.data.random_split(label_set, [len(label_set) - n_val, n_val])
    print("Initial training and validation split done!")
    # print("Training set size: ", len(train_set))
    # print("Validation set size: ", len(val_set))
    return train_set, val_set


def get_train_val_loader(train_set, val_set, batch_size=128):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    # print("Training and validation loaders done!")
    return train_loader, val_loader


def process(batch_size):
    full_dataset = load_training_data()
    label_set, unlabeled_set = rs_labeler(full_dataset)
    train_set, val_set = data_loader(label_set)
    train_loader, val_loader = get_train_val_loader(train_set, val_set, batch_size)
    return train_loader, val_loader, unlabeled_set
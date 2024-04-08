import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import pickle
import sys
sys.path.append('../')
import IndexedDataset



# Load training data
def load_train_data():
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    x_train_data = pd.read_csv(f'{data_dic_path}/x_train_time_aware_binary.csv').astype(float)
    y_train_data = pd.read_csv(f'{data_dic_path}/y_train_time_aware_binary.csv').astype(float)
    print('Training Data Loaded!')
    print('Total Length: ', len(x_train_data))
    x_train_data = torch.from_numpy(x_train_data.values).unsqueeze(1)
    y_train_data = torch.from_numpy(y_train_data.values)
    train_datast = TensorDataset(x_train_data, y_train_data)
    return train_datast

def load_test_data():
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    x_test_data = pd.read_csv(f'{data_dic_path}/x_test_time_aware_binary.csv').astype(float)
    y_test_data = pd.read_csv(f'{data_dic_path}/y_test_time_aware_binary.csv').astype(float)
    print('Testing Data Loaded!')
    print('Total Length: ', len(x_test_data))
    x_test_data = torch.from_numpy(x_test_data.values).unsqueeze(1)
    y_test_data = torch.from_numpy(y_test_data.values)
    test_datast = TensorDataset(x_test_data, y_test_data)
    return test_datast

def process(batch_size):
    train_datast = load_train_data()
    test_datast = load_test_data()
    train_loader = DataLoader(IndexedDataset.IndexedDataset(train_datast), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(IndexedDataset.IndexedDataset(test_datast), batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader
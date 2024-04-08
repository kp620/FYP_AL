import torch
import sys 
sys.path.append('../')
import Test_cuda, restnet_1d, restnet_1d_multiclass
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pickle

device, dtype = Test_cuda.check_device()

# Load the dataset
with open("../unlabel_set.pkl", 'rb') as f:
    loaded_dataset = pickle.load(f)

coreset_length = len(loaded_dataset.dataset)
print(coreset_length)
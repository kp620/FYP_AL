import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import indexed_Dataset 



# Load training data
def load_data(class_type, budget):
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    if(class_type == "binary"):
        x_data = pd.read_csv(f'{data_dic_path}/x_data_iid_binary.csv').astype(float)
        y_data = pd.read_csv(f'{data_dic_path}/y_data_iid_binary.csv').astype(float)
    elif(class_type == "multi"):
        x_data = pd.read_csv(f'{data_dic_path}/x_data_iid_multiclass.csv').astype(float)
        y_data = pd.read_csv(f'{data_dic_path}/y_data_iid_multiclass.csv').astype(float)
    print('Full Data Loaded!')
    print('Fulll Data Length: ', len(x_data))
    budget = int(len(x_data) * budget)
    x_data = torch.from_numpy(x_data.values).unsqueeze(1)
    y_data = torch.from_numpy(y_data.values)
    unique_labels = y_data.unique().tolist()
    print("Unique labels: ", len(unique_labels))
    full_dataset = TensorDataset(x_data, y_data)
    return full_dataset, budget

# Select samples to label(manually)
def rs_acquire_label(full_dataset, rs_rate = 0.05):
    num_samples = int(len(full_dataset) * rs_rate)
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    label_set = Subset(full_dataset, indices)
    all_indices = set(range(len(full_dataset)))
    chosen_indices = set(indices.tolist())
    not_chosen_indices = all_indices - chosen_indices
    unlabel_set = Subset(full_dataset, list(not_chosen_indices))
    print("Label(Training) set size: ", len(label_set))
    print("Unlabel set size: ", len(unlabel_set))
    return label_set, unlabel_set

def us_acquire_label(model, device, dtype, batch_size, full_dataset, us_rate = 0.05):
    model.eval()
    model = model.to(device=device)
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch, (input, target, idx) in enumerate(loader):
            input = input.to(device=device, dtype=dtype)
            output, _ = model(input)
            probs, _ = torch.min(torch.softmax(output, dim=1), dim=1)
            all_probs.append(probs.cpu())   
    # Concatenate all batch probabilities and sort them to find the most uncertain samples
    all_probs = torch.cat(all_probs)
    num_samples = len(all_probs)
    num_select = int(num_samples * us_rate)
    uncertainty_indices = all_probs.argsort()[:num_select]
    all_indices = set(range(len(full_dataset)))
    chosen_indices = set(uncertainty_indices.tolist())
    not_chosen_indices = all_indices - chosen_indices
    unlabel_set = Subset(full_dataset, list(not_chosen_indices))
    label_set = Subset(full_dataset, list(chosen_indices))
    print("Label(Training) set size: ", len(label_set))
    print("Unlabel set size: ", len(unlabel_set))
    return label_set, unlabel_set

# Main
def process_rs(batch_size, rs_rate, class_type, budget):
    full_dataset, budget = load_data(class_type, budget)
    label_set, unlabel_set = rs_acquire_label(full_dataset, rs_rate)
    label_loader = DataLoader(indexed_Dataset.IndexedDataset(label_set), batch_size=batch_size, shuffle=True, drop_last=False)    
    unlabel_loader = DataLoader(indexed_Dataset.IndexedDataset(unlabel_set), batch_size=batch_size, shuffle=True, drop_last=False)
    return label_loader, unlabel_loader, budget

def process_us(model, device, dtype, batch_size, rs_rate, class_type):
    full_dataset = load_data(class_type)
    label_set, unlabel_set = us_acquire_label(model, device, dtype, batch_size, full_dataset, rs_rate)
    label_loader = DataLoader(indexed_Dataset.IndexedDataset(label_set), batch_size=batch_size, shuffle=True, drop_last=False)  
    unlabel_loader = DataLoader(indexed_Dataset.IndexedDataset(unlabel_set), batch_size=batch_size, shuffle=True, drop_last=False)
    return label_loader, unlabel_loader

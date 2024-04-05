import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import pickle
import IndexedDataset



# Load training data
def load_data():
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    x_data = pd.read_csv(f'{data_dic_path}/x_data_iid_multiclass.csv').astype(float)
    y_data = pd.read_csv(f'{data_dic_path}/y_data_iid_multiclass.csv').astype(float)
    print('Training Data Loaded!')
    print('Total Length: ', len(x_data))
    x_data = torch.from_numpy(x_data.values).unsqueeze(1)
    y_data = torch.from_numpy(y_data.values)
    # unique_labels = y_data.unique().tolist()
    # print("Unique labels: ", unique_labels)
    # print("Unique labels: ", len(unique_labels))
    full_dataset = TensorDataset(x_data, y_data)
    return full_dataset

# Select samples to label(manually)
def rs_acquire_label(full_dataset, rs_rate = 0.05):
    num_samples = int(len(full_dataset) * rs_rate)
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    label_set = Subset(full_dataset, indices)
    all_indices = set(range(len(full_dataset)))
    chosen_indices = set(indices.tolist())
    not_chosen_indices = all_indices - chosen_indices

    # length_not_chosen_indices = int(len(not_chosen_indices) / 128) * 128
    # # Assuming not_chosen_indices is a set and you have a value for length_not_chosen_indices
    # not_chosen_indices_list = list(not_chosen_indices)  # Convert set to list to impose an order

    # # Now you can slice the list
    # not_chosen_indices_sliced = not_chosen_indices_list[0:length_not_chosen_indices - 1]

    # # If you need the result as a set, convert it back
    # not_chosen_indices = set(not_chosen_indices_sliced)


    unlabel_set = Subset(full_dataset, list(not_chosen_indices))
    print("label set size: ", len(label_set))
    print("unlabel set size: ", len(unlabel_set))

    return label_set, unlabel_set

def us_acquire_label(model, device, dtype, batch_size, full_dataset, us_rate = 0.05):
    model.eval()
    model = model.to(device=device)
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    all_probs = []

    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(loader):
            x_batch = x_batch.to(device=device, dtype=dtype)
            outputs, _ = model(x_batch)
            # probs, _ = torch.max(torch.softmax(outputs, dim=1), dim=1)
            probs, _ = torch.min(torch.softmax(outputs, dim=1), dim=1)
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
    print("label set size: ", len(label_set))
    print("unlabel set size: ", len(unlabel_set))

    # with open('/vol/bitbucket/kp620/FYP/us_not_chosen_indices.pkl', 'wb') as f:
    #     pickle.dump(not_chosen_indices, f)
    #     print("not_chosen_indices saved!")
    return label_set, unlabel_set

# Split the dataset into training and validation set
def train_val_split(label_set, batch_size):
    n_val = int(len(label_set) * 0.1)
    train_set, val_set = torch.utils.data.random_split(label_set, [len(label_set) - n_val, n_val])
    train_loader = DataLoader(IndexedDataset.IndexedDataset(train_set), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(IndexedDataset.IndexedDataset(val_set), batch_size=batch_size, shuffle=True)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader

# Main
def process_rs(batch_size, rs_rate):
    full_dataset = load_data()
    label_set, unlabeled_set = rs_acquire_label(full_dataset, rs_rate)
    train_loader = DataLoader(IndexedDataset.IndexedDataset(label_set), batch_size=batch_size, shuffle=True, drop_last=True)    
    unlabel_loader = DataLoader(IndexedDataset.IndexedDataset(unlabeled_set), batch_size=batch_size, shuffle=True, drop_last=True)
    # train_loader = DataLoader(label_set, batch_size=batch_size, shuffle=True)
    return train_loader, unlabel_loader

def process_us(model, device, dtype, batch_size, rs_rate):
    full_dataset = load_data()
    label_set, unlabeled_set = us_acquire_label(model, device, dtype, batch_size, full_dataset, rs_rate)
    train_loader = DataLoader(IndexedDataset.IndexedDataset(label_set), batch_size=batch_size, shuffle=True, drop_last=True)  
    unlabel_loader = DataLoader(IndexedDataset.IndexedDataset(unlabeled_set), batch_size=batch_size, shuffle=True, drop_last=True)
    # train_loader = DataLoader(label_set, batch_size=batch_size, shuffle=True)
    return train_loader, unlabel_loader


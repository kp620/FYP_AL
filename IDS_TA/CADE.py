import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from datetime import datetime
import time
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.spatial import distance
from scipy.stats import median_abs_deviation

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
device


"""
1) Main functions

"""

def load_data():
    """
    Please adjust this part of the code to suit your framework. 
    """
    return x_train, y_train, x_test, y_test



class ContrastiveDataset(Dataset):
    def __init__(self, X, y, similar_samples_ratio):
        super().__init__()
        self.X = X
        self.y = y
        self.similar_samples_ratio = similar_samples_ratio  
        self.unique_labels = np.unique(y)
        n_similar = int(similar_samples_ratio * len(y))
        n_dissimilar = len(y) - n_similar
        self.label_to_similar_indices = {label: np.where(y == label)[0] for label in self.unique_labels}
        self.label_to_dissimilar_indices = {label: np.where(y != label)[0] for label in self.unique_labels}

        self.label_to_similar_indices = {label: np.random.choice(indices, n_similar, replace=True) 
                                         for label, indices in self.label_to_similar_indices.items()}
        self.label_to_dissimilar_indices = {label: np.random.choice(indices, n_dissimilar, replace=True) 
                                            for label, indices in self.label_to_dissimilar_indices.items()}

    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        x1 = self.X[index]
        y1 = self.y[index].item()  
        should_get_similar = np.random.rand() < self.similar_samples_ratio
        if should_get_similar:
            index2 = np.random.choice(self.label_to_similar_indices[y1])
        else:
            index2 = np.random.choice(self.label_to_dissimilar_indices[y1])
        x2 = self.X[index2]
        y2 = self.y[index2].item()  
        
        return x1, y1, x2, y2

def get_dataloader(X, y, batch_size, similar_samples_ratio, shuffle=True):
    dataset = ContrastiveDataset(X, y, similar_samples_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=4)
    return dataloader




class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # encoder
        self.enc1 = nn.Linear(in_features=num_features, out_features=int(num_features*0.75))
        self.enc2 = nn.Linear(in_features=int(num_features*0.75), out_features=int(num_features*0.50))
        self.enc3 = nn.Linear(in_features=int(num_features*0.50), out_features=int(num_features*0.25))
        self.enc4 = nn.Linear(in_features=int(num_features*0.25), out_features=int(num_features*0.1))
        self.init_w()

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = self.enc4(x)
        return x
    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # decoder (reverse order of encoder)
        self.dec1 = nn.Linear(in_features=int(num_features*0.1), out_features=int(num_features*0.25))
        self.dec2 = nn.Linear(in_features=int(num_features*0.25), out_features=int(num_features*0.50))
        self.dec3 = nn.Linear(in_features=int(num_features*0.50), out_features=int(num_features*0.75))
        self.dec4 = nn.Linear(in_features=int(num_features*0.75), out_features=num_features)
        self.init_w()

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = self.dec4(x)
        return x

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def validation_loss(model, dataset, classes_centroids):
    losses = 0
    for data, target in dataset:
        projection = model(data)
        loss = lifted_structured_loss(projection, target.cuda(), alpha)
        losses += loss.item()
    return losses

def print_lr(optimizer, print_screen=True,epoch = -1):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        if print_screen == True:
            print(f'learning rate : {lr:.3f}')
    return lr


def get_lr(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def combined_loss(model,x1, x2, y1, y2, margin):
    criterion = nn.MSELoss()
    outputs1 = model(x1)
    outputs2 = model(x2)
    recon_loss = criterion(outputs1, x1) + criterion(outputs2, x2)
    dist = torch.sqrt(torch.sum((model.module.encoder(x1) - model.module.encoder(x2))**2, dim=1) + 1e-10)
    is_same = (y1 == y2).float()
    contrastive_loss = torch.mean(is_same * dist + (1 - is_same) * torch.relu(margin - dist))
    return contrastive_loss, recon_loss


def train_contrastiveAE(train_loader, model, optimizer, scheduler, epoch, margin, lambda_1, margin):
    model.train()
    train_bar = train_loader
    total_epoch_loss = 0
    lr = print_lr(optimizer, epoch)
    for batch_idx, (x1, y1, x2, y2) in enumerate(train_bar):
        x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
        optimizer.zero_grad()
        contrastive_loss, recon_loss = combined_loss(model,x1,x2, y1,y2)
        loss = lambda_1 * contrastive_loss + recon_loss
        total_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model, total_epoch_loss


def test_contrastiveAE(train_loader, model, margin, lambda_1):
    model.eval()
    train_bar = train_loader
    total_epoch_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (x1, y1, x2, y2) in enumerate(train_bar):
            x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
            outputs1 = model(x1)
            outputs2 = model(x2)
            contrastive_loss, recon_loss = combined_loss(model,x1,x2, y1, y2, margin)
            loss = lambda_1 * contrastive_loss + recon_loss
            total_epoch_loss += loss.item()
    return total_epoch_loss


def contrastiveAE_model(x_train, y_train, batch_size, similar_samples_ratio, learning_rate, margin_rate, lambda_value, valid_loader, sysname, model, optimizer):
    X = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    dataloader = get_dataloader(X, y, batch_size=batch_size, similar_samples_ratio=similar_samples_ratio)
    total_step = 250 * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max= total_step, eta_min=learning_rate * 5e-4)
    model = nn.DataParallel(model)
    best_loss = float('inf')
    for epoch in tqdm(range(1, 1+250)):
        model.train()
        model, _ = train_contrastiveAE(dataloader, model, optimizer, scheduler, epoch, margin_rate, lambda_value)
        with torch.no_grad():
            total_epoch_score = test_contrastiveAE(valid_loader, model, margin_rate, lambda_value)
            if total_epoch_score < best_loss:
                best_loss = total_epoch_score
                torch.save(model.module.state_dict(), f'{sysname}Models/CADE-m{margin_rate}-l{lambda_value}.pth')                    
    return best_loss



"""
2) Hyperparameter Search (you have to select the best model). In this case, you may try all the trained models (-_-).
"""
lmbda_list = [1.0, 0.1, 0.01, 0.001]
margin_list = [1.0, 5.0, 10.0, 15.0, 20.0]
batch_size=1024
learning_rate = 0.0001
similar_samples_ratio = 0.25
x_val = torch.tensor(x_valid, dtype=torch.float32)
y_val = torch.tensor(y_valid, dtype=torch.long)

val_dataloader = get_dataloader(x_val, y_val, batch_size=batch_size, similar_samples_ratio=similar_samples_ratio)

for lmbda in lmbda_list:
    for margin in margin_list:
        model = Autoencoder().cuda()
        optimizer_fn = torch.optim.Adam
        optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
        start_time = time.time()
        print(f'Start time: {time.strftime("%H:%M:%S", time.gmtime(start_time))}')
        score = contrastiveAE_model(x_train, y_train, batch_size, similar_samples_ratio, learning_rate, margin, lmbda, val_dataloader, chosen_name, model, optimizer)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        elapsed_time_hours = elapsed_time_minutes / 60
        print(f'End time: {time.strftime("%H:%M:%S", time.gmtime(end_time))}')
        print(f'Elapsed time for lambda= {lmbda}, margin= {margin}: {elapsed_time_seconds} seconds, {elapsed_time_minutes} minutes, {elapsed_time_hours} hours, loss {score}')


"""
3) Prediction and Selection Function


"""


def predict(x_test, model):
    model.eval()
    x_test = torch.tensor(x_test, dtype=torch.float32)
    x_test = x_test.cuda()
    if isinstance(model, torch.nn.DataParallel):
        z_test = model.module.encoder(x_test)
    else:
        z_test = model.encoder(x_test)
    return z_test.detach().cpu().numpy()

z_train = predict(x_train, model)
print(z_train.shape)

test_latent = predict(x_test, model)

def get_latent_data_for_each_family(z_train, y_train):
    N = len(np.unique(y_train))
    N_family = [len(np.where(y_train == family)[0]) for family in range(N)]
    z_family = []
    for family in range(N):
        z_tmp = z_train[np.where(y_train == family)[0]]
        z_family.append(z_tmp)
    z_len = [len(z_family[i]) for i in range(N)]
    print(f'z_family length: {z_len}')

    return N, N_family, z_family

def get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family):
    dis_family = []  
    for i in range(N): 
        dis = [np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(N_family[i])]
        dis_family.append(dis)
    dis_len = [len(dis_family[i]) for i in range(N)]
    print(f'dis_family length: {dis_len}')

    return dis_family

def get_MAD_for_each_family(dis_family, N, N_family):
    mad_family = []
    median_list = []
    for i in range(N):
        median = np.median(dis_family[i])
        median_list.append(median)
        print(f'family {i} median: {median}')
        diff_list = [np.abs(dis_family[i][j] - median) for j in range(N_family[i])]
        mad = 1.4826 * np.median(diff_list)  # 1.4826: assuming the underlying distribution is Gaussian
        mad_family.append(mad)
    print(f'mad_family: {mad_family}')

    return mad_family, median_list

N, N_family, z_family = get_latent_data_for_each_family(z_train, y_train)
centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
print(f'centroids: {centroids}')
dis_family = get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family)
mad_family, dis_family = get_MAD_for_each_family(dis_family, N, N_family)


def sample_selection(X_test, y_test, z_test, thres, selection_rate=0.01):
    centroids_array = np.array(centroids)  
    dis_matrix = distance.cdist(z_test, centroids_array, 'euclidean')
    
    dis_k_minus_dis_family = dis_matrix - np.array(dis_family)
    anomaly_k = np.abs(dis_k_minus_dis_family) / np.array(mad_family)
    min_anomaly_score = np.min(anomaly_k, axis=1)
    sorted_indexes_asc = np.argsort(min_anomaly_score)
    sorted_indexes_desc = sorted_indexes_asc[::-1]
    sr = int(selection_rate * len(X_test))
    selected_idx = sorted_indexes_desc[:sr]
    return X_test[selected_idx], y_test[selected_idx]

selected_x, selected_y = sample_selection(x_train, y_train, z_train, 3.5, selection_rate=0.01)



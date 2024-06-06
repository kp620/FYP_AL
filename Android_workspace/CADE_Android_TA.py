import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import restnet_1d_android as resnet_1d
import model_Trainer_Android as model_Trainer
import test_Cuda
import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import indexed_Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from datetime import datetime
import time
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.spatial import distance
from scipy.stats import median_abs_deviation
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn import init
import glob

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)



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
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        num_features = 1159
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
        num_features = 1159
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
    
class main_tariner():
    def __init__(self):

        self.Model_trainer = model_Trainer # Model trainer
        self.device, self.dtype = test_Cuda.check_device()
        self.CADE_model = Autoencoder().to(device=self.device, dtype=self.dtype)


        self.classifer_model = resnet_1d.build_model("multi")
        self.classifer_model_batch_size = 1024 # Batch size
        self.classifer_model_lr = 0.00001 # Learning rate
        self.classifer_model_optimizer = optim.Adam(
            self.classifer_model.parameters(),
            lr=self.classifer_model_lr, weight_decay=0.0001
        )
        self.classifer_model_criterion = nn.CrossEntropyLoss() # Loss function

        self.selection_rate = 400

        self.lmbda_list = [1.0, 0.1, 0.01, 0.001]
        self.margin_list = [1.0, 5.0, 10.0, 15.0, 20.0]
        self.similar_samples_ratio = 0.25

        self.CADE_optimizer_fn = torch.optim.Adam
        self.CADE_optimizer = self.CADE_optimizer_fn(self.CADE_model.parameters(), lr=0.0001)

        
        self.directory = '/vol/bitbucket/kp620/FYP/NON_FINAL/Android_workspace/data/gen_apigraph_drebin'
        self.training_file = [f'{self.directory}/2012-01to2012-12_selected.npz']
        # self.validation_file = [
        #     f'{self.directory}/2013-01_selected.npz',
        #     f'{self.directory}/2013-02_selected.npz',
        #     f'{self.directory}/2013-03_selected.npz',
        #     f'{self.directory}/2013-04_selected.npz',
        #     f'{self.directory}/2013-05_selected.npz',
        #     f'{self.directory}/2013-06_selected.npz',
        #     ]
        # self.test_files = [
        #     f'{self.directory}/2013-07_selected.npz',
        #     f'{self.directory}/2013-08_selected.npz',
        #     f'{self.directory}/2013-09_selected.npz',
        #     f'{self.directory}/2013-10_selected.npz',
        #     f'{self.directory}/2013-11_selected.npz',
        #     f'{self.directory}/2013-12_selected.npz',
        # ]
        # self.test_files.extend(glob.glob(f'{self.directory}/2014*.npz'))
        # self.test_files.extend(glob.glob(f'{self.directory}/2015*.npz'))
        # self.test_files.extend(glob.glob(f'{self.directory}/2016*.npz'))
        # self.test_files.extend(glob.glob(f'{self.directory}/2017*.npz'))
        # self.test_files.extend(glob.glob(f'{self.directory}/2018*.npz'))

        # file_to_remove = f'{self.directory}/2017-11_selected.npz'
        # self.test_files.remove(file_to_remove)
        # self.test_files = sorted(self.test_files)
        # self.validation_files = sorted(self.validation_file)
        self.test_files = []
        self.test_files.extend(glob.glob(f'{self.directory}/2013*.npz'))
        self.test_files.extend(glob.glob(f'{self.directory}/2014*.npz'))
        self.test_files.extend(glob.glob(f'{self.directory}/2015*.npz'))
        self.test_files.extend(glob.glob(f'{self.directory}/2016*.npz'))
        self.test_files.extend(glob.glob(f'{self.directory}/2017*.npz'))
        self.test_files.extend(glob.glob(f'{self.directory}/2018*.npz'))
        file_to_remove = f'{self.directory}/2017-11_selected.npz'
        self.test_files.remove(file_to_remove)
        self.test_files = sorted(self.test_files)


        self.X_train = None
        self.y_train = None

    def load_file(self, file):
        data = np.load(file)
        x_train = data['X_train']
        y_train = data['y_train']

        y_train_binary = np.where(y_train == 0, 0, 1) # Convert to binary

        y_mal_family = data['y_mal_family']
        return x_train, y_train_binary, y_mal_family

    def load_data(self, file):
        """
        Please adjust this part of the code to suit your framework. 
        """
        x_train = []
        y_train = []
        y_mal_family = []
        command = "echo 'Loading file: " + str(file) + "'"
        subprocess.call(command, shell=True)
        x, y, y_mal = self.load_file(os.path.join(self.directory, file))
        x_train.append(x)
        y_train.append(y)
        y_mal_family.append(y_mal)
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        x_data = pd.DataFrame(x_train).astype(float)
        y_data = pd.DataFrame(y_train).astype(float)
        command = "echo 'length of full data: " + str(len(x_data)) + "'"
        subprocess.call(command, shell=True)

        x_data = torch.from_numpy(x_data.values).unsqueeze(1)
        y_data = torch.from_numpy(y_data.values)
        
        return x_data, y_data

    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_dataloader(self, X, y, batch_size, similar_samples_ratio, shuffle=True):
        dataset = ContrastiveDataset(X, y, similar_samples_ratio)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=4)
        return dataloader
    
    def print_lr(self, optimizer, print_screen=True,epoch = -1):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            if print_screen == True:
                print(f'learning rate : {lr:.3f}')
        return lr

    def combined_loss(self, model,x1, x2, y1, y2, margin):
        criterion = nn.MSELoss()
        outputs1 = model(x1)
        outputs2 = model(x2)
        recon_loss = criterion(outputs1, x1) + criterion(outputs2, x2)
        # dist = torch.sqrt(torch.sum((model.module.encoder(x1) - model.module.encoder(x2))**2, dim=1) + 1e-10)
        encoded_x1 = model.module.encoder(x1) if isinstance(model, nn.DataParallel) else model.encoder(x1)
        encoded_x2 = model.module.encoder(x2) if isinstance(model, nn.DataParallel) else model.encoder(x2)
        
        dist = torch.sqrt(torch.sum((encoded_x1 - encoded_x2) ** 2, dim=1) + 1e-10)
        is_same = (y1 == y2).float().unsqueeze(1)
        # print(f"is_same shape: {is_same.shape}")
        # print(f"dist shape: {dist.shape}")
        contrastive_loss = torch.mean(is_same * dist + (1 - is_same) * torch.relu(margin - dist))
        return contrastive_loss, recon_loss
    
    def train_contrastiveAE(self, train_loader, model, optimizer, scheduler, epoch, margin, lambda_1):
        model.train()
        train_bar = train_loader
        total_epoch_loss = 0
        lr = self.print_lr(optimizer, epoch)
        for batch_idx, (x1, y1, x2, y2) in enumerate(train_bar):
            x1, y1, x2, y2 = x1.float().cuda(), y1.cuda(), x2.float().cuda(), y2.cuda()
            optimizer.zero_grad()
            contrastive_loss, recon_loss = self.combined_loss(model,x1,x2, y1,y2, margin)
            loss = lambda_1 * contrastive_loss + recon_loss
            total_epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        return model, total_epoch_loss
    
    def test_contrastiveAE(self, train_loader, model, margin, lambda_1):
        model.eval()
        train_bar = train_loader
        total_epoch_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch_idx, (x1, y1, x2, y2) in enumerate(train_bar):
                x1, y1, x2, y2 = x1.float().cuda(), y1.cuda(), x2.float().cuda(), y2.cuda()
                outputs1 = model(x1)
                outputs2 = model(x2)
                contrastive_loss, recon_loss = self.combined_loss(model,x1,x2, y1, y2, margin)
                loss = lambda_1 * contrastive_loss + recon_loss
                total_epoch_loss += loss.item()
        return total_epoch_loss

    def contrastiveAE_model(self, x_train, y_train, batch_size, similar_samples_ratio, learning_rate, margin_rate, lambda_value, model, optimizer):
        X = torch.tensor(x_train, dtype=torch.float32)
        y = torch.tensor(y_train, dtype=torch.long)
        dataloader = self.get_dataloader(X, y, batch_size=batch_size, similar_samples_ratio=similar_samples_ratio)
        total_step = 250 * len(dataloader)
        scheduler = CosineAnnealingLR(optimizer, T_max= total_step, eta_min=learning_rate * 5e-4)
        model = nn.DataParallel(model)
        for epoch in tqdm(range(1, 1+250)):
            model.train()
            model, _ = self.train_contrastiveAE(dataloader, model, optimizer, scheduler, epoch, margin_rate, lambda_value)

    def predict(self, x_test, model):
        model.eval()
        x_test = torch.tensor(x_test, dtype=torch.float32)
        x_test = x_test.cuda()
        if isinstance(model, torch.nn.DataParallel):
            z_test = model.module.encoder(x_test)
        else:
            z_test = model.encoder(x_test)
        return z_test.detach().cpu().numpy()

    def get_latent_data_for_each_family(self, z_train, y_train):
        N = len(np.unique(y_train))
        N_family = [len(np.where(y_train == family)[0]) for family in range(N)]
        z_family = []
        for family in range(N):
            z_tmp = z_train[np.where(y_train == family)[0]]
            z_family.append(z_tmp)
        z_len = [len(z_family[i]) for i in range(N)]
        print(f'z_family length: {z_len}')

        return N, N_family, z_family

    def get_latent_distance_between_sample_and_centroid(self, z_family, centroids, N, N_family):
        dis_family = []  
        for i in range(N): 
            dis = [np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(N_family[i])]
            dis_family.append(dis)
        dis_len = [len(dis_family[i]) for i in range(N)]
        print(f'dis_family length: {dis_len}')

        return dis_family

    def get_MAD_for_each_family(self, dis_family, N, N_family):
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

    def assign_labels(self, z_test, centroids):
        dist_matrix = distance.cdist(z_test, centroids, 'euclidean')
        labels = np.argmin(dist_matrix, axis=1)
        return labels

    def sample_selection(self, X_test, y_test, z_test, centroids, dis_family, mad_family):
        sr = self.selection_rate
        centroids_array = np.array(centroids) 
        # print("centroids_array shape: ", centroids_array.shape)
        # print("z_test shape: ", z_test.shape)
        dis_matrix = distance.cdist(z_test, centroids_array, 'euclidean')
        
        dis_k_minus_dis_family = dis_matrix - np.array(dis_family)
        anomaly_k = np.abs(dis_k_minus_dis_family) / np.array(mad_family)
        min_anomaly_score = np.min(anomaly_k, axis=1)
        sorted_indexes_asc = np.argsort(min_anomaly_score)
        sorted_indexes_desc = sorted_indexes_asc[::-1]
        selected_idx = sorted_indexes_desc[:sr]

        selected_idx_copy = selected_idx.copy().astype(int)
        return selected_idx_copy
    

    def initial_train(self):
        print("Train 2012 start")
        X_train , y_train = self.load_data(self.training_file[0])
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(indexed_Dataset.IndexedDataset(train_dataset), batch_size=1024, shuffle=True, drop_last=False)
        self.Model_trainer.initial_train(train_loader, self.classifer_model, self.device, self.dtype, self.classifer_model_criterion, self.classifer_model_lr)
        print("Train 2012 complete")


    def main(self):
        for l in [0.1]:
            for m in [5.0]:
                learning_rate = 0.0001
                self.initial_train()
                
                self.CADE_model = Autoencoder().to(device=self.device, dtype=self.dtype)

                self.X_train, self.y_train = self.load_data(self.training_file[0])
                self.X_train = self.X_train.reshape(-1, 1159)
                self.contrastiveAE_model(self.X_train, self.y_train, 1024, 0.25, learning_rate, m, l, self.CADE_model, self.CADE_optimizer)
                
                z_train = self.predict(self.X_train, self.CADE_model)
                N, N_family, z_family = self.get_latent_data_for_each_family(z_train, self.y_train)
                centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
                dis_family = self.get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family)
                mad_family, dis_family = self.get_MAD_for_each_family(dis_family, N, N_family)

                for i in range(0,len(self.test_files)-1):
                    # X_data, y_data = self.load_data(file)
                    file = self.test_files[i]
                    print("train file: ", file)
                    X_data, y_test = self.load_data(file)
                    X_test = X_data.reshape(-1, 1159)

                    selected_indices = self.sample_selection(X_test, y_test, self.predict(X_test, self.CADE_model), centroids, dis_family, mad_family)
                    selected_indices_torch = torch.tensor(selected_indices, dtype=torch.long)
                    x_selected = X_test[selected_indices_torch.numpy()]
                    y_selected = y_test[selected_indices_torch.numpy()]

                    self.X_train = np.concatenate((self.X_train, x_selected))
                    self.y_train = np.concatenate((self.y_train, y_selected))
                    self.contrastiveAE_model(self.X_train, self.y_train, self.classifer_model_batch_size, self.similar_samples_ratio, learning_rate, m, l, self.CADE_model, self.CADE_optimizer)
                    

                    # ----------------------------
                    # Train the classifier  
                    
                    classifier_x_train = X_data[selected_indices]
                    classifier_y_train = y_test[selected_indices]
                    classifier_train_dataset = TensorDataset(torch.tensor(classifier_x_train, dtype=torch.float32), torch.tensor(classifier_y_train, dtype=torch.long))
                    classifier_train_loader = DataLoader(indexed_Dataset.IndexedDataset(classifier_train_dataset), batch_size=self.classifer_model_batch_size, shuffle=True, drop_last=False)
                    self.Model_trainer.initial_train(classifier_train_loader, self.classifer_model, self.device, self.dtype, self.classifer_model_criterion, self.classifer_model_lr)
                    # ----------------------------


                    z_train = self.predict(self.X_train, self.CADE_model)
                    N, N_family, z_family = self.get_latent_data_for_each_family(z_train, self.y_train)
                    centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
                    dis_family = self.get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family)
                    mad_family, dis_family = self.get_MAD_for_each_family(dis_family, N, N_family)


                    # ----------------------------
                    # Test the classifier
                    print("Testing the classifier on next month")
                    test_file = self.test_files[i+1]
                    print("Test file: ", test_file)
                    X_test, y_test = self.load_data(test_file)
                    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
                    test_loader = DataLoader(indexed_Dataset.IndexedDataset(test_dataset), batch_size=self.classifer_model_batch_size, shuffle=False, drop_last=False)
                    predictions = []
                    targets = []
                    with torch.no_grad():
                        for batch, (input, target, idx) in enumerate(test_loader):
                            input = input.to(self.device, dtype=self.dtype)
                            target = target.to(self.device, dtype=self.dtype).squeeze().long()

                            output, _ = self.classifer_model(input)
                            probabilities = F.softmax(output, dim=1)
                            _, pseudo_label = torch.max(probabilities, dim=1)
                            predictions.extend(pseudo_label.cpu().numpy())
                            targets.extend(target.cpu().numpy())

                    predictions = np.array(predictions).astype(int)
                    targets = np.array(targets).astype(int)

                    unique_labels = np.unique(targets).tolist()
                    print(f'{file} Conservative classification Report for m={m}, l={l}: ')
                    print(classification_report(targets, predictions, labels=unique_labels, zero_division=0))
                    print(f'{file} Non-conservative classification Report for m={m}, l={l}: ')
                    print(classification_report(targets, predictions, labels=unique_labels, zero_division=1))
                


trainer = main_tariner()
trainer.main()
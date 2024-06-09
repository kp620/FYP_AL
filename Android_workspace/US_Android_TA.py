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


    
class main_tariner():
    def __init__(self):

        self.Model_trainer = model_Trainer # Model trainer
        self.device, self.dtype = test_Cuda.check_device()


        self.classifer_model = resnet_1d.build_model("multi")
        self.classifer_model_batch_size = 1024 # Batch size
        self.classifer_model_lr = 0.00001 # Learning rate
        self.classifer_model_optimizer = optim.Adam(
            self.classifer_model.parameters(),
            lr=self.classifer_model_lr, weight_decay=0.0001
        )
        self.classifer_model_criterion = nn.CrossEntropyLoss() # Loss function

        self.selection_rate = 400

        
        self.directory = '/vol/bitbucket/kp620/FYP/NON_FINAL/Android_workspace/data/gen_apigraph_drebin'
        self.training_file = [f'{self.directory}/2012-01to2012-12_selected.npz']
        
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

    
    def uncertainty_sampling(self, model, x_data, y_data, num_select, device, dtype, batch_size = 1024):
        num_select = int(num_select)
        # Make sure the model is in evaluation mode
        model.eval()
        # Ensure x_data and y_data are on the same device as the model
        x_data = x_data.to(device, dtype = dtype)
        y_data = y_data.to(device, dtype = dtype).long()
        model = model.to(device=device)
        dataset = TensorDataset(x_data, y_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for batch_idx, (x_batch, _) in enumerate(loader):
                x_batch = x_batch.to(device=device, dtype=dtype)
                output, _ = model(x_batch)
                probs, _ = torch.max(torch.softmax(output, dim=1), dim=1)
                all_probs.append(probs.cpu())   
        # Concatenate all batch probabilities and sort them to find the most uncertain samples
        all_probs = torch.cat(all_probs)
        uncertainty_indices = all_probs.argsort()[:num_select]
        model.train()
        return uncertainty_indices


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
                self.initial_train()

                for i in range(0,len(self.test_files)-1):
                    file = self.test_files[i]
                    print("train file: ", file)
                    X_data, y_test = self.load_data(file)


                    for _ in range(10):
                        selected_indices = self.uncertainty_sampling(self.classifer_model, X_data, y_test, self.selection_rate/5, self.device, self.dtype)
                        # ----------------------------
                        # Train the classifier  
                        classifier_x_train = X_data[selected_indices]
                        classifier_y_train = y_test[selected_indices]
                        classifier_train_dataset = TensorDataset(torch.tensor(classifier_x_train, dtype=torch.float32), torch.tensor(classifier_y_train, dtype=torch.long))
                        classifier_train_loader = DataLoader(indexed_Dataset.IndexedDataset(classifier_train_dataset), batch_size=self.classifer_model_batch_size, shuffle=True, drop_last=False)
                        self.Model_trainer.initial_train(classifier_train_loader, self.classifer_model, self.device, self.dtype, self.classifer_model_criterion, self.classifer_model_lr)
                        # ----------------------------

                        all_indices = torch.arange(X_data.shape[0])
                        not_selected_indices = all_indices[~torch.isin(all_indices, selected_indices)]
                        X_data = X_data[not_selected_indices]
                        y_test = y_test[not_selected_indices]


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
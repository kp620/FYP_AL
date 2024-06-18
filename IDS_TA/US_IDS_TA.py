"""
Code performing US on the IDS dataset under TA setting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import restnet_1d_dynamic as resnet_1d
import model_Trainer_ta as model_Trainer
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

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


class DynamicClassifier(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(DynamicClassifier, self).__init__()
        self.linear = nn.Linear(feature_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
    def get_weights_size(self):
        return self.linear.weight.size()

    def get_biases_size(self):
        return self.linear.bias.size()
    
    def get_weights(self):
        return self.linear.weight

    def get_biases(self):
        return self.linear.bias
    
    def update_classifier(self, old_weights, old_biases, new_num_classes, original_num_classes):
        """
        Updates the classifier to handle more classes, transferring old weights and biases,
        and initializing new weights and biases using Kaiming initialization.
        """
        # Current feature size and number of original classes
        original_num_classes = original_num_classes
        feature_size = self.linear.in_features

        # Create new linear layer with more classes
        self.linear = nn.Linear(feature_size, new_num_classes)

        # Transfer old weights and biases
        if old_weights is not None and old_biases is not None:
            self.linear.weight.data[:original_num_classes, :] = old_weights
            self.linear.bias.data[:original_num_classes] = old_biases

        # Initialize the weights for the new classes with Kaiming initialization
        if new_num_classes > original_num_classes:
            init.kaiming_normal_(self.linear.weight.data[original_num_classes:, :], mode='fan_out', nonlinearity='relu')
            self.linear.bias.data[original_num_classes:].fill_(0)  # Initialize new biases to zero

        new_weights_size = self.linear.weight.size()
        new_biases_size = self.linear.bias.size()
        command = "echo 'New weight size: " + str(new_weights_size) + "'"
        subprocess.call(command, shell=True)
        command = "echo 'New bias size: " + str(new_biases_size) + "'"
        subprocess.call(command, shell=True)

    
class main_tariner():
    def __init__(self):

        self.Model_trainer = model_Trainer # Model trainer
        self.device, self.dtype = test_Cuda.check_device()

        self.num_classes = 3
        self.dynamic_classifier = DynamicClassifier(feature_size=512, num_classes=self.num_classes)
        self.dynamic_classifier.to(self.device)

        self.classifer_model = resnet_1d.build_model("multiclass")
        self.classifer_model_batch_size = 1024 # Batch size
        self.classifer_model_lr = 0.00001 # Learning rate
        self.classifer_model_optimizer = optim.Adam(
            list(self.classifer_model.parameters()) + list(self.dynamic_classifier.parameters()),
            lr=self.classifer_model_lr, weight_decay=0.0001
        )
        self.classifer_model_criterion = nn.CrossEntropyLoss() # Loss function

        self.selection_rate = 0.03

        self.directory = '/vol/bitbucket/kp620/FYP/dataset'
        self.Monday_file = f'{self.directory}/Monday-WorkingHours.csv'
        self.Tuesday_file = f'{self.directory}/Tuesday-WorkingHours.csv'
        self.Wednesday_file = f'{self.directory}/Wednesday-WorkingHours.csv'
        self.Thursday_file = f'{self.directory}/Thursday-WorkingHours.csv'
        self.Friday_file = f'{self.directory}/Friday-WorkingHours.csv'

        self.full_x_data = f'{self.directory}/x_data_iid_multiclass.csv'
        self.full_y_data = f'{self.directory}/y_data_iid_multiclass.csv'

        self.Mon_Tue_data , self.Mon_Tue_x, self.Mon_Tue_y = None, None, None
        self.Wednesday_data, self.Wednesday_x, self.Wednesday_y = None, None, None
        self.Thursday_data, self.Thursday_x, self.Thursday_y = None, None, None

        self.dictionary = {
            "0":0, 
            "1":1,
            "2":2
            }

    def load_data(self):
        """
        Please adjust this part of the code to suit your framework. 
        """
        x_data = pd.read_csv(self.full_x_data).astype(float)
        y_data = pd.read_csv(self.full_y_data).astype(float)
        x_data = torch.from_numpy(x_data.values).unsqueeze(1)
        y_data = torch.from_numpy(y_data.values)
        
        return x_data, y_data

    def uncertainty_sampling(self, model, dynamic_classifier, x_data, y_data, selection_rate, device, dtype, batch_size = 1024):
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
                output = model(x_batch)
                output = dynamic_classifier(output)
                probs, _ = torch.max(torch.softmax(output, dim=1), dim=1)
                all_probs.append(probs.cpu())   
        # Concatenate all batch probabilities and sort them to find the most uncertain samples
        all_probs = torch.cat(all_probs)
        num_samples = len(all_probs)
        num_select = int(num_samples * selection_rate)
        uncertainty_indices = all_probs.argsort()[:num_select]
        return uncertainty_indices

    
    def get_data(self):
        command = 'echo "Loading data"'
        subprocess.call(command, shell=True)
        x_data, y_data = self.load_data()
        self.Mon_Tue_data = x_data[:len(pd.read_csv(self.Monday_file)) + len(pd.read_csv(self.Tuesday_file)), :]
        self.Mon_Tue_x = self.Mon_Tue_data.reshape(-1, 78)
        self.Mon_Tue_y = y_data[:len(pd.read_csv(self.Monday_file)) + len(pd.read_csv(self.Tuesday_file)), :]

        Wednesday_start = len(pd.read_csv(self.Monday_file)) + len(pd.read_csv(self.Tuesday_file))
        Wednesday_end = Wednesday_start + len(pd.read_csv(self.Wednesday_file))
        self.Wednesday_data = x_data[Wednesday_start:Wednesday_end, :]
        self.Wednesday_x = self.Wednesday_data.reshape(-1, 78)
        self.Wednesday_y = y_data[Wednesday_start:Wednesday_end, :]

        Thursday_start = Wednesday_end
        Thursday_end = Thursday_start + len(pd.read_csv(self.Thursday_file))
        self.Thursday_data = x_data[Thursday_start:Thursday_end, :]
        self.Thursday_x = self.Thursday_data.reshape(-1, 78)
        self.Thursday_y = y_data[Thursday_start:Thursday_end, :]

        Friday_start = Thursday_end
        Friday_end = Friday_start + len(pd.read_csv(self.Friday_file))
        self.Friday_data = x_data[Friday_start:Friday_end, :]
        self.Friday_x = self.Friday_data.reshape(-1, 78)
        self.Friday_y = y_data[Friday_start:Friday_end, :]

    def MONTUE_train(self):
        command = 'echo "Training on Monday and Tuesday data start"'
        subprocess.call(command, shell=True)
        train_dataset = TensorDataset(self.Mon_Tue_data, self.Mon_Tue_y)
        train_loader = DataLoader(indexed_Dataset.IndexedDataset(train_dataset), batch_size=1024, shuffle=True, drop_last=False)
        self.Model_trainer.initial_train(train_loader, self.classifer_model, self.device, self.dtype, self.classifer_model_criterion, self.classifer_model_lr, self.dynamic_classifier, self.classifer_model_optimizer)
        command = 'echo "Training on Monday and Tuesday data end"'
        subprocess.call(command, shell=True)

    def change_num_classes(self, num_classes_new):
        if num_classes_new > self.num_classes:
            command = "echo 'Old number of classes: " + str(self.num_classes) + "'"
            subprocess.call(command, shell=True)
            command = "echo 'Changing number of classes to " + str(num_classes_new) + "'"
            subprocess.call(command, shell=True)
            old_weights = self.dynamic_classifier.get_weights()
            old_biases = self.dynamic_classifier.get_biases()
            self.dynamic_classifier.update_classifier(old_weights, old_biases, num_classes_new, self.num_classes)
            self.num_classes = num_classes_new
            self.dynamic_classifier.to(self.device)
            self.classifer_model_optimizer.param_groups.clear()
            self.classifer_model_optimizer.add_param_group({'params': self.dynamic_classifier.parameters()})
            self.classifer_model_optimizer.add_param_group({'params': self.classifer_model.parameters()})
            self.dynamic_classifier.to(self.device)
        else:
            command = "echo 'Number of classes is the same!'"
            subprocess.call(command, shell=True)

    def load_model_states(self, resnet_model, dynamic_classifier):
        # Load the dictionary of state dicts from the file
        state_dicts = torch.load(f'{self.directory}/initial_model.pth')

        # Load state dicts into each model
        resnet_model.load_state_dict(state_dicts['resnet_model_state_dict'])
        dynamic_classifier.load_state_dict(state_dicts['dynamic_classifier_state_dict'])

    def main(self):
        self.get_data()

        self.load_model_states(self.classifer_model, self.dynamic_classifier)
        for day in ['Wednesday', 'Thursday', 'Friday']:
            if day == 'Wednesday':
                X_data = self.Wednesday_data
                y_data = self.Wednesday_y
            elif day == 'Thursday':
                X_data = self.Thursday_data
                y_data = self.Thursday_y
            elif day == 'Friday':
                X_data = self.Friday_data
                y_data = self.Friday_y


            for epoch in range(0,10):
                command = f'echo "Epoch {epoch+1}"'
                subprocess.call(command, shell=True)
                selected_indices = self.uncertainty_sampling(self.classifer_model, self.dynamic_classifier, X_data, y_data, self.selection_rate, self.device, self.dtype)

                # ----------------------------
                # Train the classifier
                command = 'echo "Training the classifier"'
                subprocess.call(command, shell=True)
                selected_labels = []
                dictionary_keys = {int(k) for k in self.dictionary.keys()}
                
                for idx in selected_indices:
                    selected_labels.append(y_data[idx])
                
                unique_labels = np.unique(selected_labels)
                unique_labels = {int(l) for l in selected_labels}

                command = "echo 'Unique labels: " + str(unique_labels) + "'"
                subprocess.call(command, shell=True)
                command = "echo 'Dictionary keys: " + str(dictionary_keys) + "'"
                subprocess.call(command, shell=True)

                labels_not_in_dictionary = unique_labels - dictionary_keys
                if len(labels_not_in_dictionary) > 0:
                    for label in labels_not_in_dictionary:
                        self.dictionary[str(int(label))] = int(len(self.dictionary))
                        command = "echo 'Label: " + str(label) + " added to dictionary!'"
                        subprocess.call(command, shell=True)
                command = "echo 'Corresponding value: " + str(self.dictionary[str(int(label))]) + "'"
                subprocess.call(command, shell=True)
                command = "echo 'Dictionary: " + str(self.dictionary) + "'"
                subprocess.call(command, shell=True)

                self.change_num_classes(len(self.dictionary))
            
                classifier_x_train = X_data[selected_indices]
                classifier_y_train = [self.dictionary[str(int(label.item()))] for label in y_data[selected_indices]]
                classifier_train_dataset = TensorDataset(torch.tensor(classifier_x_train, dtype=torch.float32), torch.tensor(classifier_y_train, dtype=torch.long))
                classifier_train_loader = DataLoader(indexed_Dataset.IndexedDataset(classifier_train_dataset), batch_size=self.classifer_model_batch_size, shuffle=True, drop_last=False)
                self.Model_trainer.initial_train(classifier_train_loader, self.classifer_model, self.device, self.dtype, self.classifer_model_criterion, self.classifer_model_lr, self.dynamic_classifier, self.classifer_model_optimizer)
                # ----------------------------
                all_indices = torch.arange(X_data.shape[0])
                not_selected_indices = all_indices[~torch.isin(all_indices, selected_indices)]
                X_data = X_data[not_selected_indices]
                y_data = y_data[not_selected_indices]

            # ----------------------------
            # Test the classifier
            command = 'echo "Testing the classifier"'
            subprocess.call(command, shell=True)
            X_test = X_data
            y_test = y_data
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
            test_loader = DataLoader(indexed_Dataset.IndexedDataset(test_dataset), batch_size=self.classifer_model_batch_size, shuffle=False, drop_last=False)
            predictions = []
            targets = []
            with torch.no_grad():
                for batch, (input, target, idx) in enumerate(test_loader):
                    input = input.to(self.device, dtype=self.dtype)
                    target = target.to(self.device, dtype=self.dtype).long()

                    target = torch.tensor([
                        self.dictionary[str(int(t.item()))] if str(int(t.item())) in self.dictionary else t + 100
                        for t in target.cpu()
                    ], dtype=self.dtype, device=self.device).long()

                    output = self.classifer_model(input)
                    output = self.dynamic_classifier(output)
                    probabilities = F.softmax(output, dim=1)
                    _, pseudo_label = torch.max(probabilities, dim=1)
                    predictions.extend(pseudo_label.cpu().numpy())
                    targets.extend(target.cpu().numpy())

            predictions = np.array(predictions).astype(int)
            targets = np.array(targets).astype(int)

            unique_labels = np.unique(targets).tolist()
            print(f'{day} Conservative classification Report: ')
            print(classification_report(targets, predictions, labels=unique_labels, zero_division=0))
            print(f'{day} Non-conservative classification Report: ')
            print(classification_report(targets, predictions, labels=unique_labels, zero_division=1))

                


trainer = main_tariner()
trainer.main()
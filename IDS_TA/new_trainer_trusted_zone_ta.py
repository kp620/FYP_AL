"""
Main training logic for the coreset selection algorithm with trust region constraint under TA setting.
"""

# --------------------------------
# Import area
import test_Cuda, approx_Optimizer, facility_Update, indexed_Dataset, uncertainty_similarity
import model_Trainer_ta as model_Trainer
import restnet_1d_dynamic as restnet_1d
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import torch.nn.functional as F
import subprocess
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch.nn.init as init

# --------------------------------
# Argument parser
# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument("--operation_type",  choices=['iid'], help='Specify operation type: "iid"')
parser.add_argument("--class_type", choices=['multi'], help='Specify class type: "multi"')
parser.add_argument("--budget", type=float, help='Specify the budget ratio')
args = parser.parse_args()

# Define the dynamic classifier, which determines the number of classes of the output layer
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


# --------------------------------
# Main processing logic
class main_trainer():
    def __init__(self, args=args):
        self.args = args
        # Device & Data type
        self.device, self.dtype = test_Cuda.check_device() # Check if cuda is available

        # Model & Parameters
        self.Model_trainer = model_Trainer # Model trainer
        self.Model = restnet_1d # Model

        self.num_classes = 3 # Number of classes
        self.dynamic_classifier = DynamicClassifier(feature_size=512, num_classes=self.num_classes)
        self.dynamic_classifier.to(self.device) 

        self.model = self.Model.build_model(class_type=self.args.class_type) # Model used to train the data(M_0)
        self.batch_size = 1024 # Batch size
        self.lr = 0.00001 # Learning rate
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001) # Optimizer
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.dynamic_classifier.parameters()),
            lr=self.lr, weight_decay=0.0001
        )
        self.criterion = nn.CrossEntropyLoss() # Loss function

        # Counter
        self.steps_per_epoch = int
        self.reset_step = int
        self.train_iter = iter
        
        # Variables
        self.label_loader = DataLoader # Labelled dataset
        self.unlabel_loader = DataLoader # Unlabelled dataset
        self.train_loader = DataLoader # Training dataset
        self.approx_loader = DataLoader # Approximation dataset
        self.coreset_loader = DataLoader # Coreset dataset
        self.delta = 1.0 
        self.gf = 0 
        self.ggf = 0
        self.ggf_moment = 0
        self.start_loss = 0
        # self.rs_rate = 0.001 # In IID, the rate of random sampling that is used to train the initial model
        self.budget_ratio = self.args.budget # Budget
        self.budget = 0
        self.gradients = [] # Gradients of the unlabel set
        self.pseudo_labels = [] # Pseudo labels of the unlabel set
        self.subsets = []
        self.coreset_index = []
        self.weights = []
        self.train_weights = []
        self.final_weights = []
        self.final_coreset = None
        self.gradient_approx_optimizer = approx_Optimizer.Adahessian(list(self.model.parameters()) + list(self.dynamic_classifier.parameters()))
        self.stop = 0
        self.alpha = 0.25
        self.alpha_max = 0.75
        self.sigma = 0.4
        self.eigenv = None
        self.optimal_step = None

        self.directory = "/vol/bitbucket/kp620/FYP/dataset"
        self.Monday_file = f'{self.directory}/Monday-WorkingHours.csv'
        self.Tuesday_file = f'{self.directory}/Tuesday-WorkingHours.csv'
        self.Wednesday_file = f'{self.directory}/Wednesday-WorkingHours.csv'
        self.Thursday_file = f'{self.directory}/Thursday-WorkingHours.csv'
        self.Friday_file = f'{self.directory}/Friday-WorkingHours.csv'
        self.full_x_data = f'{self.directory}/x_data_iid_multiclass.csv'
        self.full_y_data = f'{self.directory}/y_data_iid_multiclass.csv'
        
        # Dictionary to store the labels, the initial labels are 0, 1, 2 because we only train on Monday and Tuesday
        self.dictionary = {
            "0":0, 
            "1":1,
            "2":2
            }
        
        self.eigenv = np.load(f'{self.directory}/eigvecs_d.npy') # Load eigenvectors
        self.eigenv = torch.tensor(self.eigenv, dtype=self.dtype, device=self.device)
        self.train_length = 0 # Length of the training data
        self.full_dataset = TensorDataset # Full dataset

        
    # Change the number of classes in the model
    def change_num_classes(self, num_classes_new):
        if num_classes_new > self.num_classes:
            # need to change the number of classes in the model without reinitializing the model
            print("Changing number of classes...")
            print("Old number of classes: ", self.num_classes)
            print("Changing number of classes to ", num_classes_new)
            old_weights = self.dynamic_classifier.get_weights()
            old_biases = self.dynamic_classifier.get_biases()
            
            # Update the classifier
            self.dynamic_classifier.update_classifier(old_weights, old_biases, num_classes_new, self.num_classes)
            self.num_classes = num_classes_new
            self.dynamic_classifier.to(self.device)
            
            # Update the optimizer
            self.optimizer.param_groups.clear()
            self.optimizer.add_param_group({'params': self.model.parameters()})
            self.optimizer.add_param_group({'params': self.dynamic_classifier.parameters()})
            self.gradient_approx_optimizer = approx_Optimizer.Adahessian(list(self.model.parameters()) + list(self.dynamic_classifier.parameters()))
            self.dynamic_classifier.to(self.device)
        else: 
            print("Number of classes is the same!")
        

    def load_full_data(self):
        print("Loading full data...")
        x_data = pd.read_csv(self.full_x_data).astype(float)
        y_data = pd.read_csv(self.full_y_data).astype(float)
        x_data = torch.from_numpy(x_data.values)
        y_data = torch.from_numpy(y_data.values)
        full_dataset = TensorDataset(x_data, y_data)
        x_data = full_dataset.tensors[0].numpy()
        x_data = torch.from_numpy(x_data).unsqueeze(1)
        self.full_dataset = TensorDataset(x_data, full_dataset.tensors[1])
    
    # Train the model on Monday and Tuesday data
    def train_MONTUE(self):
        print("Training on Monday and Tuesday data...")
        train_length = len(pd.read_csv(self.Monday_file)) + len(pd.read_csv(self.Tuesday_file))
        train_dataset = Subset(self.full_dataset, np.arange(self.train_length, self.train_length + train_length))
        self.train_length += train_length
        self.label_loader = DataLoader(indexed_Dataset.IndexedDataset(train_dataset), batch_size=self.batch_size, shuffle=True, drop_last=False)
        print("Initial training started...")
        self.Model_trainer.initial_train(self.label_loader, self.model, self.device, self.dtype, criterion=self.criterion, learning_rate=self.lr, classifer=self.dynamic_classifier, optimizer=self.optimizer)
        print("Initial training complete!")


    def load_unlabel_data(self, file):
        train_length = len(pd.read_csv(file))
        unlabel_dataset = Subset(self.full_dataset, np.arange(self.train_length, min(self.train_length + train_length, len(self.full_dataset))))
        self.train_length += train_length
        self.unlabel_loader = DataLoader(indexed_Dataset.IndexedDataset(unlabel_dataset), batch_size=self.batch_size, shuffle=True, drop_last=False)

        # Acquire pseudo labels of the unlabel set
        # ---------------------print begin---------------------
        print("Acquiring pseudo labels...")
        self.pseudo_labels = self.Model_trainer.psuedo_labeling(self.model, self.device, self.dtype, loader=self.unlabel_loader, classifer=self.dynamic_classifier)
        print("Pseudo labels acquired!")
        # ---------------------print end---------------------

        self.steps_per_epoch = np.ceil(int(len(self.unlabel_loader.dataset) * self.budget_ratio) / self.batch_size).astype(int)
        # ---------------------print begin---------------------
        print("Steps per epoch: ", self.steps_per_epoch)
        # ---------------------print end---------------------

         # ---------------------print begin---------------------
        self.budget += int(train_length * self.budget_ratio)
        print("Budget: ", self.budget)
        # ---------------------print end---------------------

    # Redistribute the labels to align with the chronological order
    def redistribute_label(self):
        print("Redistributing labels...")
        selected_labels = []
        coreset_loader = DataLoader(Subset(self.unlabel_loader.dataset, indices=self.coreset_index), batch_size=self.batch_size, shuffle=False, drop_last=False)
        for batch, (input, target, idx) in enumerate(coreset_loader):
            target = target.to(self.device, dtype=self.dtype).squeeze().long()
            selected_labels.extend(target.cpu().numpy())
        selected_labels = np.unique(selected_labels)

        dictionary_keys = {int(k) for k in self.dictionary.keys()}
        unique_labels = {int(l) for l in selected_labels}

        print("Unique labels: ", unique_labels)
        print("Dictionary keys: ", dictionary_keys)
        labels_not_in_dictionary = unique_labels - dictionary_keys
        if len(labels_not_in_dictionary) > 0:
            print("New labels found!")
            for label in labels_not_in_dictionary:
                self.dictionary[str(label)] = int(len(self.dictionary))
                print("Label: ", label, " added to dictionary!")
                print("Corresponding value: ", self.dictionary[str(label)])
        print("Dictionary: ", self.dictionary)
        print("Labels redistributed!")
        
 
    def train_epoch(self, epoch):
        print("Training epoch: ", epoch)
        self.reset_step = self.steps_per_epoch
        # Training loop
        self.model.train()
        for training_step in range(self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)):
            print("Training step: ", training_step)
            # Check the approximation error
            if((training_step >= self.reset_step) and ((training_step - self.reset_step) % self.steps_per_epoch == 0)): 
                print("updating trustzone at step: ", training_step)
                self.update_trustzone(training_step)
                print("Trustzone updated!")

            # Update the coreset
            if training_step == self.reset_step or training_step == 0:
                print("Updating coreset at step: ", training_step)

                continuous_state = uncertainty_similarity.continuous_states(self.eigenv, self.label_loader, self.unlabel_loader, self.model, self.device, self.dtype, alpha=self.alpha, sigma=self.sigma, classifer=self.dynamic_classifier)
                self.alpha = min(self.alpha + 0.01, self.alpha_max)
                continuous_state = continuous_state[:, None]

                self.gradients = self.Model_trainer.gradient_train(training_step, self.model, self.unlabel_loader, self.pseudo_labels, self.device, self.dtype, batch_size=self.batch_size, criterion=self.criterion, classifer=self.dynamic_classifier)
                self.gradients = self.gradients * continuous_state
                

                print("Selecting subset at step: ", training_step)
                self.select_subset(training_step)
                print("Subset selected!")

                # Given the coreset index, we need to redistritbute the labels
                self.redistribute_label()
                self.change_num_classes(len(self.dictionary))
                
                for weight in self.weights:
                    self.final_weights.append(weight)
                
                self.update_train_loader_and_weights(training_step)

                # Update the train loader and weights
                if self.final_coreset is None:
                    self.final_coreset = Subset(self.unlabel_loader.dataset, self.coreset_index)
                else:
                    if(len(self.final_coreset) > self.budget):
                        print("Budget reached!")
                        self.stop = 1
                        break
                    self.final_coreset = ConcatDataset([self.final_coreset, Subset(self.unlabel_loader.dataset, self.coreset_index)])
                print("Final coreset length at step: ", training_step, " is ", len(self.final_coreset))
                
                self.label_loader = ConcatDataset([self.label_loader.dataset.dataset, Subset(self.unlabel_loader.dataset.dataset, self.coreset_index)])
                self.label_loader = DataLoader(indexed_Dataset.IndexedDataset(self.label_loader), batch_size=self.batch_size, shuffle=True, drop_last=False)
                self.train_loader = self.coreset_loader
                self.train_iter = iter(self.train_loader)
                print("Quadratic approximation at step: ", training_step)
                self.get_quadratic_approximation()
                print("Quadratic approximation complete!")
            try: 
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            data, target, idx = batch
            data, target = data.to(self.device, dtype=self.dtype), target.to(self.device, dtype=self.dtype).squeeze().long()

            target = torch.tensor([self.dictionary[str(t.item())] for t in target.cpu()], dtype=self.dtype, device=self.device).long()
            print("Forward and backward at step: ", training_step)
            self.forward_and_backward(data, target, idx)
            print("Forward and backward complete!")

    def save_model_states(self, resnet_model, dynamic_classifier):
        # Save both state dictionaries in a single file
        torch.save({
            'resnet_model_state_dict': resnet_model.state_dict(),
            'dynamic_classifier_state_dict': dynamic_classifier.state_dict()
        }, f'{self.directory}/initial_model.pth')

    def load_model_states(self, resnet_model, dynamic_classifier):
        # Load the dictionary of state dicts from the file
        state_dicts = torch.load(f'{self.directory}/initial_model.pth')

        # Load state dicts into each model
        resnet_model.load_state_dict(state_dicts['resnet_model_state_dict'])
        dynamic_classifier.load_state_dict(state_dicts['dynamic_classifier_state_dict'])
    
    def main_train(self):
        self.load_full_data()
        # self.train_MONTUE()
        # # save the weights of the linear layer 
        # self.save_model_states(self.model, self.dynamic_classifier)
        # weights_size = self.dynamic_classifier.get_weights_size()
        # biases_size = self.dynamic_classifier.get_biases_size()

        # print("Weights Size:", weights_size)  # Example Output: torch.Size([3, 512])
        # print("Biases Size:", biases_size)    # Example Output: torch.Size([3])

        self.load_model_states(self.model, self.dynamic_classifier)
        print("Model loaded!")

        train_length = len(pd.read_csv(self.Monday_file)) + len(pd.read_csv(self.Tuesday_file))
        train_dataset = Subset(self.full_dataset, np.arange(self.train_length, self.train_length + train_length))
        self.train_length += train_length
        total_data_points = len(train_dataset)
        subset_size = int(0.01 * total_data_points)
        random_indices = torch.randperm(total_data_points)[:subset_size]
        subset_dataset = Subset(train_dataset, random_indices)
        self.label_loader = DataLoader(
            indexed_Dataset.IndexedDataset(subset_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        for day in range(3, 6):
            if day == 3:
                file = self.Wednesday_file
                print("Loading Wednesday data...")
            elif day == 4:
                self.stop = 0 
                file = self.Thursday_file
                print("Loading Thursday data...")
            elif day == 5:
                self.stop = 0
                file = self.Friday_file
                print("Loading Friday data...")
            self.load_unlabel_data(file)
            for epoch in range(100):
                self.train_epoch(epoch)
                if self.stop == 1:
                    break
            self.test_accuracy()
                


# --------------------------------
# Auxiliary functions     
    def apply_fractional_optimal_step(self):
        # Apply a fraction of the optimal step
        fraction = self.optimal_step / self.steps_per_epoch
        combined_params = list(self.model.parameters()) + list(self.dynamic_classifier.parameters())
        with torch.no_grad():
            for param, fr in zip(combined_params, fraction):
                param -= fr

    def forward_and_backward(self, data, target, idx):
        self.optimizer.zero_grad()
        output = self.model(data)
        output = self.dynamic_classifier(output)
        loss = self.criterion(output, target)
        loss = (loss * self.train_weights[idx]).mean()
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.apply_fractional_optimal_step()

    def update_trustzone(self, training_step):   
        true_loss = 0 
        count = 0 
        self.approx_loader = DataLoader(Subset(self.unlabel_loader.dataset, indices=self.coreset_index), batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.model.eval()
        self.dynamic_classifier.eval()
        for batch, (data, target, idx) in enumerate(self.approx_loader):
            data = data.to(self.device, dtype=self.dtype)
            # pseudo_y = self.pseudo_labels[idx].to(self.device, dtype=self.dtype).squeeze().long()
            target = target.to(self.device, dtype=self.dtype).squeeze().long()

            pseudo_y = torch.tensor([self.dictionary[str(t.item())] for t in target.cpu()], dtype=self.dtype, device=self.device).long()

            output = self.model(data)
            output = self.dynamic_classifier(output)
            loss = self.criterion(output, pseudo_y)
            loss.backward()
            true_loss += loss.item() * data.size(0)
            count += data.size(0)
        self.model.train()
        self.dynamic_classifier.train()

        true_loss = true_loss / count
        approx_loss = torch.matmul(self.optimal_step, self.gf) + self.start_loss
        approx_loss += 1/2 * torch.matmul(self.optimal_step * self.ggf, self.optimal_step)
        
        actual_reduction = self.start_loss - true_loss
        approx_reduction = self.start_loss - approx_loss
        rho = actual_reduction / approx_reduction
        if rho > 0.75:
            self.delta *= 2  # Expand the trust region
        elif rho < 0.1:
            self.delta *= 0.5  # Contract the trust region

        self.reset_step = training_step
        all_indices = set(range(len(self.unlabel_loader.dataset)))
        coreset_indices = set(self.coreset_index)
        remaining_indices = list(all_indices - coreset_indices)
        unlabel_set = self.unlabel_loader.dataset.dataset
        unlabel_set = Subset(unlabel_set, remaining_indices)
        self.unlabel_loader = DataLoader(indexed_Dataset.IndexedDataset(unlabel_set), batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.pseudo_labels = self.Model_trainer.psuedo_labeling(self.model, self.device, self.dtype, loader=self.unlabel_loader, classifer=self.dynamic_classifier)
            
    
    def get_quadratic_approximation(self):
        # second-order approximation with coreset
        self.approx_loader = DataLoader(Subset(self.unlabel_loader.dataset, indices=self.coreset_index), batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.start_loss = 0 
        count = 0 
        for batch, (input, target, idx) in enumerate (self.approx_loader):
            input = input.to(self.device, dtype=self.dtype)
            # pseudo_y = self.pseudo_labels[idx].to(self.device, dtype=self.dtype).squeeze().long()
            target = target.to(self.device, dtype=self.dtype).squeeze().long()

            pseudo_y = torch.tensor([self.dictionary[str(t.item())] for t in target.cpu()], dtype=self.dtype, device=self.device).long()

            output = self.model(input)
            output = self.dynamic_classifier(output)
            loss = self.criterion(output, pseudo_y)
            batch_weight = self.train_weights[idx.long()]
            loss = (loss * batch_weight).mean()
            self.model.zero_grad()
            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(momentum=True)
            if batch == 0:
                self.gf = gf_tmp * len(idx)
                self.ggf = ggf_tmp * len(idx)
                self.ggf_moment = ggf_tmp_moment * len(idx)
            else:
                self.gf += gf_tmp * len(idx)
                self.ggf += ggf_tmp * len(idx)
                self.ggf_moment += ggf_tmp_moment * len(idx)
            self.start_loss += loss.item() * input.size(0) # each batch contributes to the total loss proportional to its size
            count += input.size(0)

        self.gf /= len(self.approx_loader.dataset)
        self.ggf /= len(self.approx_loader.dataset)
        self.ggf_moment /= len(self.approx_loader.dataset)
        self.start_loss = self.start_loss / count 
        
        # Calculate step using the trust region constraint
        diag_H_inv = 1.0 / self.ggf  # Element-wise inverse of the Hessian diagonal approximation
        step = -diag_H_inv * self.gf
        step_norm = torch.norm(step)
        if step_norm > self.delta:
            step *= self.delta / step_norm  # Scale step to fit within the trust region
        self.optimal_step = step

    def update_train_loader_and_weights(self, training_step):
        print("Updating train loader and weights at step: ", training_step)
        self.coreset_loader = DataLoader(Subset(self.unlabel_loader.dataset, indices=self.coreset_index), batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.train_weights = np.zeros(len(self.unlabel_loader.dataset))
        self.weights = self.weights / np.sum(self.weights) * len(self.coreset_index)
        self.train_weights[self.coreset_index] = self.weights
        self.train_weights = torch.from_numpy(self.train_weights).float().to(self.device)

    def select_random_set(self):
        all_indices = np.arange(len(self.gradients))
        indices = []
        for c in np.unique(self.pseudo_labels):
            class_indices = np.intersect1d(np.where(self.pseudo_labels == c)[0], all_indices)
            indices_per_class = np.random.choice(class_indices, size=min(max(int(np.ceil(0.001 * len(class_indices))),1),len(class_indices)), replace=False)
            indices.append(indices_per_class)
        indices = np.concatenate(indices).astype(int)
        return indices

    def select_subset(self, training_step):
        subsets = []
        self.coreset_index = []
        self.subsets = []
        self.weights = []
        for iteration in range(10):
            random_subset = self.select_random_set()
            subsets.append(random_subset)
        # Greedy Facility Location
        print("Greedy FL Start at step: ", training_step)
        subset_count = 0
        for subset in subsets: 
            if subset_count % 1 == 0:
                print("Handling subset #", subset_count, " out of #", len(subsets))
            gradient_data = self.gradients[subset]
            # if np.shape(gradient_data)[-1] == 2:
            #      gradient_data -= np.eye(2)[self.pseudo_labels[subset]] 
            # gradient_data = self.gradients[subset].squeeze()
            if gradient_data.size <= 0:
                continue
            fl_labels = self.pseudo_labels[subset] - torch.min(self.pseudo_labels[subset]) # Ensure the labels start from 0
            sub_coreset_index, sub_weights= facility_Update.get_orders_and_weights(128, gradient_data, "euclidean", y=fl_labels.cpu().numpy(), equal_num=False, mode="sparse", num_n=128)
            sub_coreset_index = subset[sub_coreset_index] # Get the indices of the coreset
            self.coreset_index.extend(sub_coreset_index.tolist()) # Add the coreset to the coreset list
            self.weights.extend(sub_weights.tolist()) # Add the weights to the weights list
            self.subsets.extend(subset)
            subset_count += 1
        print("Greedy FL Complete at step: ", training_step)

    def test_accuracy(self):
        print("Testing accuracy...")
        unlabel_loader = self.unlabel_loader
        
        self.model.eval()
        self.dynamic_classifier.eval()
        predictions = []
        targets = []
        # Disable gradient calculations
        with torch.no_grad():
            for batch, (input, target, idx) in enumerate(unlabel_loader):
                input = input.to(self.device, dtype=self.dtype)
                target = target.to(self.device, dtype=self.dtype).squeeze().long()

                # target = torch.tensor([self.dictionary[str(t.item())] for t in target.cpu()], dtype=self.dtype, device=self.device).long()
                target = torch.tensor([
                    self.dictionary[str(t.item())] if str(t.item()) in self.dictionary else t + 100
                    for t in target.cpu()
                ], dtype=self.dtype, device=self.device).long()

                output = self.model(input)
                output = self.dynamic_classifier(output)
                probabilities = F.softmax(output, dim=1)
                _, pseudo_label = torch.max(probabilities, dim=1)
                predictions.extend(pseudo_label.cpu().numpy())
                targets.extend(target.cpu().numpy())
        predictions = np.array(predictions).astype(int)
        targets = np.array(targets).astype(int)

        unique_labels = np.unique(targets).tolist()
        print("Conservative classification Report: ")
        print(classification_report(targets, predictions, labels=unique_labels, zero_division=0))
        print("Non-Conservative classification Report: ")
        print(classification_report(targets, predictions, labels=unique_labels, zero_division=1))


        self.model.train()
        self.dynamic_classifier.train()



caller = main_trainer(args=args)
caller.main_train()
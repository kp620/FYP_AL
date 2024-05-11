# --------------------------------
# Import area
import test_Cuda, approx_Optimizer, indexed_Dataset, uncertainty_similarity
import restnet_1d_android as restnet_1d
import facility_Update_Android as facility_Update
import model_Trainer_Android as model_Trainer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import torch.nn.functional as F
import subprocess
import pandas as pd
import argparse
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --------------------------------
# Argument parser
# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument("--operation_type",  choices=['iid', 'time-aware'], help='Specify operation type: "iid" or "time-aware"')
parser.add_argument("--class_type", choices=['binary', 'multi'], help='Specify class type: "binary" or "multi"')
parser.add_argument("--budget", type=float, help='Specify the budget')
args = parser.parse_args()


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
        self.model = self.Model.build_model(class_type=self.args.class_type) # Model used to train the data(M_0)
        self.batch_size = 128 # Batch size
        self.lr = 0.00001 # Learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001) # Optimizer
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
        self.budget = 0 # Budget
        self.check_thresh_factor = 0.1
        self.delta = 1.0 
        self.gf = 0 
        self.ggf = 0
        self.ggf_moment = 0
        self.start_loss = 0
        self.budget = self.args.budget * 6 # Budget
        self.gradients = [] # Gradients of the unlabel set
        self.pseudo_labels = [] # Pseudo labels of the unlabel set
        self.subsets = []
        self.coreset_index = []
        self.weights = []
        self.train_weights = []
        self.final_weights = []
        self.final_coreset = None
        self.gradient_approx_optimizer = approx_Optimizer.Adahessian(self.model.parameters())
        self.stop = 0
        self.alpha = 0.25
        self.alpha_max = 0.75
        self.sigma = 0.4
        self.eigenv = None
        self.optimal_step = None

    def load_data(self,file):
        data = np.load(file)
        x_train = data['X_train']
        y_train = data['y_train']
        y_mal_family = data['y_mal_family']
        return x_train, y_train, y_mal_family
    
    def load_month(self,month_file):
        # x_train = np.load(month_file)['X_train']
        # y_train = np.load(month_file)['y_train']
        # y_mal_family = np.load(month_file)['y_mal_family']
        x_train, y_train, y_mal_family = self.load_data(month_file)
        x_data = pd.DataFrame(x_train).astype(float)
        y_data = pd.DataFrame(y_train).astype(float)
        x_data = torch.from_numpy(x_data.values)
        y_data = torch.from_numpy(y_data.values)
        dataset = TensorDataset(x_data, y_data)
        x_data = dataset.tensors[0].numpy()
        x_data = torch.from_numpy(x_data).unsqueeze(1)
        dataset = TensorDataset(x_data, dataset.tensors[1])
        return dataset
    

    def initial_training(self, month_file):
        command = "echo 'Loading(Initial training) month file: " + str(month_file) + "'"
        subprocess.call(command, shell=True)
        train_dataset = self.load_month(month_file)
        self.label_loader = DataLoader(indexed_Dataset.IndexedDataset(train_dataset), batch_size=self.batch_size, shuffle=True, drop_last=False)
    
        x_unlabel = []
        y_unlabel = []
        y_mal_family_unlabel = []
        directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_apigraph_drebin'
        validation_files = [
            f'{directory}/2013-01_selected.npz',
            f'{directory}/2013-02_selected.npz',
            f'{directory}/2013-03_selected.npz',
            f'{directory}/2013-04_selected.npz',
            f'{directory}/2013-05_selected.npz',
            f'{directory}/2013-06_selected.npz',
        ]
        for files in validation_files:
            command = "echo 'Loading unlabel file: " + str(files) + "'"
            subprocess.call(command, shell=True)
            x, y, y_mal = self.load_data(files)
            x_unlabel.append(x)
            y_unlabel.append(y)
            y_mal_family_unlabel.append(y_mal)
        x_unlabel = np.concatenate(x_unlabel)
        y_unlabel = np.concatenate(y_unlabel)

        x_unlabel = pd.DataFrame(x_unlabel).astype(float)
        y_unlabel = pd.DataFrame(y_unlabel).astype(float)
        x_unlabel = torch.from_numpy(x_unlabel.values)
        y_unlabel = torch.from_numpy(y_unlabel.values)
        unlabel_dataset = TensorDataset(x_unlabel, y_unlabel)
        x_unlabel = unlabel_dataset.tensors[0].numpy()
        x_unlabel = torch.from_numpy(x_unlabel).unsqueeze(1)
        unlabel_dataset = TensorDataset(x_unlabel, unlabel_dataset.tensors[1])
        self.unlabel_loader = DataLoader(indexed_Dataset.IndexedDataset(unlabel_dataset), batch_size=self.batch_size, shuffle=True, drop_last=False)


        command = "echo 'Initial training...'"
        subprocess.call(command, shell=True)
        self.Model_trainer.initial_train(self.label_loader, self.model, self.device, self.dtype, criterion=self.criterion, learning_rate=self.lr)
        command = "echo 'Initial training complete!'"
        subprocess.call(command, shell=True)
        self.label_loader = None

        # Acquire pseudo labels of the unlabel set
        # ---------------------print begin---------------------
        command = "echo 'Acquiring pseudo labels...'"
        subprocess.call(command, shell=True)
        self.pseudo_labels = self.Model_trainer.psuedo_labeling(self.model, self.device, self.dtype, loader=self.unlabel_loader)
        command = "echo 'Pseudo labels acquired!'"
        subprocess.call(command, shell=True)
        # ---------------------print end---------------------

        self.steps_per_epoch = np.ceil(int(len(self.unlabel_loader.dataset) * 0.1) / self.batch_size).astype(int)
        # ---------------------print begin---------------------
        command = "echo 'Steps per epoch: " + str(self.steps_per_epoch) + "'" 
        subprocess.call(command, shell=True)
        # ---------------------print end---------------------
        print("Steps per epoch: ", self.steps_per_epoch)

 
    def train_epoch(self, epoch):
        print("Training epoch: ", epoch)
        self.reset_step = self.steps_per_epoch
        # Training loop
        self.model.train()
        for training_step in range(self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)):
            # ---------------------print begin---------------------
            command = "echo 'Training step: " + str(training_step) + "'"
            subprocess.call(command, shell=True)
            # ---------------------print end---------------------
            # Check the approximation error
            if((training_step >= self.reset_step) and ((training_step - self.reset_step) % self.steps_per_epoch == 0)): 
                # ---------------------print begin---------------------
                command = "echo 'Extend coreset at step: " + str(training_step) + "'"
                subprocess.call(command, shell=True)
                self.extend_coreset(training_step)
                command = "echo 'Extend coreset complete!'"
                subprocess.call(command, shell=True)
                # ---------------------print end---------------------

            # Update the coreset
            if training_step == self.reset_step or training_step == 0:
                # print("Updating coreset at step: ", training_step)
                # ---------------------print begin---------------------
                command = "echo 'Updating coreset at step: " + str(training_step) + "'"
                subprocess.call(command, shell=True)
                # ---------------------print end---------------------

                # ---------------------print begin---------------------
                # continuous_state = uncertainty_similarity.continuous_states(self.eigenv, self.label_loader, self.unlabel_loader, self.model, self.device, self.dtype, alpha=self.alpha, sigma=self.sigma)
                # self.alpha = min(self.alpha + 0.01, self.alpha_max)
                # continuous_state = continuous_state[:, None]
                # command = "echo 'Continuous state shape: " + str(len(continuous_state)) + "'"
                # subprocess.call(command, shell=True)


                self.gradients = self.Model_trainer.gradient_train(training_step, self.model, self.unlabel_loader, self.pseudo_labels, self.device, self.dtype, batch_size=self.batch_size, criterion=self.criterion)
                command = "echo 'Gradients shape: " + str(len(self.gradients)) + "'"
                subprocess.call(command, shell=True)

                # self.gradients = self.gradients * continuous_state
                # ---------------------print end---------------------
                
                
                # ---------------------print begin---------------------
                command = "echo 'select_subset at step: " + str(training_step) + "'"
                subprocess.call(command, shell=True)
                self.select_subset(training_step)
                command = "echo 'select_subset complete!'"
                subprocess.call(command, shell=True)
                # ---------------------print end---------------------
                
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
                # ---------------------print begin---------------------
                command = "echo 'Final coreset length at step: " + str(training_step) + " is " + str(len(self.final_coreset)) + "'"
                subprocess.call(command, shell=True)
                # ---------------------print end---------------------
                

                if self.label_loader is None:
                    self.label_loader = Subset(self.unlabel_loader.dataset.dataset, self.coreset_index)
                else:
                    self.label_loader = ConcatDataset([self.label_loader.dataset.dataset, Subset(self.unlabel_loader.dataset.dataset, self.coreset_index)])
                self.label_loader = DataLoader(indexed_Dataset.IndexedDataset(self.label_loader), batch_size=self.batch_size, shuffle=True, drop_last=False)
                self.train_loader = self.coreset_loader
                self.train_iter = iter(self.train_loader)
                # ---------------------print begin---------------------
                command = "echo 'Quadratic approximation at step: " + str(training_step) + "'"
                subprocess.call(command, shell=True)
                self.get_quadratic_approximation()
                command = "echo 'Quadratic approximation complete!'"
                subprocess.call(command, shell=True)
                # ---------------------print end---------------------
            try: 
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            data, target, idx = batch
            data, target = data.to(self.device, dtype=self.dtype), target.to(self.device, dtype=self.dtype).squeeze().long()
            # ---------------------print begin---------------------
            command = "echo 'forward_and_backward at step: " + str(training_step) + "'"
            subprocess.call(command, shell=True)
            self.forward_and_backward(data, target, idx)
            command = "echo 'forward_and_backward at step: " + str(training_step) + " complete!'"
            subprocess.call(command, shell=True)
            # ---------------------print end---------------------
    
    def main_train(self, epoch):
        directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_apigraph_drebin'
        train_directory = f'{directory}/2012-01to2012-12_selected.npz'
        self.initial_training(train_directory)

        self.random_test()
        return

        for e in range(epoch):
            command = "echo 'Epoch: " + str(e) + "'"
            subprocess.call(command, shell=True)
            self.train_epoch(e)
            if self.stop == 1:
                break
        self.test_accuracy_without_weight()
        self.test_accuracy_with_weight()

# --------------------------------
# Auxiliary functions     
    def apply_fractional_optimal_step(self):
        # Apply a fraction of the optimal step
        fraction = self.optimal_step / self.steps_per_epoch
        with torch.no_grad():
            for param, fr in zip(self.model.parameters(), fraction):
                param -= fr

    def forward_and_backward(self, data, target, idx):
        self.optimizer.zero_grad()
        output, _ = self.model(data)
        loss = self.criterion(output, target)
        loss = (loss * self.train_weights[idx]).mean()
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.apply_fractional_optimal_step()

    def extend_coreset(self, training_step):   
        true_loss = 0 
        count = 0 
        self.approx_loader = DataLoader(Subset(self.unlabel_loader.dataset, indices=self.coreset_index), batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.model.eval()
        for batch, (data, target, idx) in enumerate(self.approx_loader):
            data = data.to(self.device, dtype=self.dtype)
            # pseudo_y = self.pseudo_labels[idx].to(self.device, dtype=self.dtype).squeeze().long()
            pseudo_y = target.to(self.device, dtype=self.dtype).squeeze().long()
            output,_ = self.model(data)
            loss = self.criterion(output, pseudo_y)
            loss.backward()
            true_loss += loss.item() * data.size(0)
            count += data.size(0)
        self.model.train()

        true_loss = true_loss / count
        approx_loss = torch.matmul(self.optimal_step, self.gf) + self.start_loss
        approx_loss += 1/2 * torch.matmul(self.optimal_step * self.ggf, self.optimal_step)
        
        actual_reduction = self.start_loss - true_loss
        approx_reduction = self.start_loss - approx_loss
        rho = actual_reduction / approx_reduction
        command = "echo 'Rho at step: " + str(training_step) + " is " + str(rho) + "'"
        subprocess.call(command, shell=True)
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
        self.pseudo_labels = self.Model_trainer.psuedo_labeling(self.model, self.device, self.dtype, loader=self.unlabel_loader)
            
    
    def get_quadratic_approximation(self):
        # second-order approximation with coreset
        self.approx_loader = DataLoader(Subset(self.unlabel_loader.dataset, indices=self.coreset_index), batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.start_loss = 0 
        count = 0 
        for batch, (input, target, idx) in enumerate (self.approx_loader):
            input = input.to(self.device, dtype=self.dtype)
            # pseudo_y = self.pseudo_labels[idx].to(self.device, dtype=self.dtype).squeeze().long()
            pseudo_y = target.to(self.device, dtype=self.dtype).squeeze().long()
            output, _ = self.model(input)
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
        print("Quadratic approximation complete!")

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
            indices_per_class = np.random.choice(class_indices, size=int(np.ceil(0.001 * len(class_indices))), replace=False)
            indices.append(indices_per_class)
        indices = np.concatenate(indices).astype(int)
        return indices

    def select_subset(self, training_step):
        subsets = []
        self.coreset_index = []
        self.subsets = []
        self.weights = []
        for iteration in range(2):
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
            sub_coreset_index, sub_weights= facility_Update.get_orders_and_weights(25, gradient_data, "euclidean", y=fl_labels.cpu().numpy(), equal_num=False, mode="sparse", num_n=128)
            sub_coreset_index = subset[sub_coreset_index] # Get the indices of the coreset
            self.coreset_index.extend(sub_coreset_index.tolist()) # Add the coreset to the coreset list
            self.weights.extend(sub_weights.tolist()) # Add the weights to the weights list
            self.subsets.extend(subset)
            subset_count += 1
        print("Greedy FL Complete at step: ", training_step)

    def load_test_data(self, test_files):
        command = "echo 'Loading unlabel file: " + str(test_files) + "'"
        subprocess.call(command, shell=True)
        x_unlabel = []
        y_unlabel = []
        x, y, y_mal = self.load_data(test_files)
        x_unlabel.append(x)
        y_unlabel.append(y)
        x_unlabel = np.concatenate(x_unlabel)
        y_unlabel = np.concatenate(y_unlabel)

        x_unlabel = pd.DataFrame(x_unlabel).astype(float)
        y_unlabel = pd.DataFrame(y_unlabel).astype(float)
        x_unlabel = torch.from_numpy(x_unlabel.values)
        y_unlabel = torch.from_numpy(y_unlabel.values)
        unlabel_dataset = TensorDataset(x_unlabel, y_unlabel)
        x_unlabel = unlabel_dataset.tensors[0].numpy()
        x_unlabel = torch.from_numpy(x_unlabel).unsqueeze(1)
        unlabel_dataset = TensorDataset(x_unlabel, unlabel_dataset.tensors[1])
        unlabel_loader = DataLoader(indexed_Dataset.IndexedDataset(unlabel_dataset), batch_size=self.batch_size, shuffle=True, drop_last=False)
        return unlabel_loader


    def test_accuracy_without_weight(self):
        command = "echo 'Testing accuracy without weight!'"
        subprocess.call(command, shell=True)
        test_model = self.Model.build_model(class_type=self.args.class_type)
        optimizer = optim.Adam(test_model.parameters(), lr=self.lr, weight_decay=0.0001)
        print("Testing accuracy without weight!")
        coreset = self.final_coreset
        print("len coreset: ", len(coreset))
        coreset_loader = DataLoader(coreset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # Training loop
        test_model.train()
        test_model = test_model.to(device=self.device)
        num_epochs = 100
        for epoch in range(num_epochs):
            for batch,(input, target, idx) in enumerate(coreset_loader):
                input = input.to(device=self.device, dtype=self.dtype)
                target = target.to(device=self.device, dtype=self.dtype).squeeze().long()
                optimizer.zero_grad()
                output, _ = test_model(input)
                loss = self.criterion(output,target)
                loss.backward()
                optimizer.step()  
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_apigraph_drebin'
        test_files = [
            f'{directory}/2013-07_selected.npz',
            f'{directory}/2013-08_selected.npz',
            f'{directory}/2013-09_selected.npz',
            f'{directory}/2013-10_selected.npz',
            f'{directory}/2013-11_selected.npz',
            f'{directory}/2013-12_selected.npz'
        ]
        test_files.extend(glob.glob(f'{directory}/2014*.npz'))
        test_files.extend(glob.glob(f'{directory}/2015*.npz'))
        test_files.extend(glob.glob(f'{directory}/2016*.npz'))
        test_files.extend(glob.glob(f'{directory}/2017*.npz'))
        test_files.extend(glob.glob(f'{directory}/2018*.npz'))

        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        count = 0

        for files in test_files:
            unlabel_loader = self.load_test_data(files)
            test_model.eval()
            predictions = []
            targets = []
            # Disable gradient calculations
            with torch.no_grad():
                for batch, (input, target, idx) in enumerate(unlabel_loader):
                    input = input.to(self.device, dtype=self.dtype)
                    target = target.to(self.device, dtype=self.dtype).squeeze().long()
                    output, _ = test_model(input)
                    probabilities = F.softmax(output, dim=1)
                    _, pseudo_label = torch.max(probabilities, dim=1)
                    predictions.extend(pseudo_label.cpu().numpy())
                    targets.extend(target.cpu().numpy())
            predictions = np.array(predictions)
            targets = np.array(targets)
            accuracy = accuracy_score(targets, predictions)
            if(self.args.class_type == 'binary'):
                average = "binary"
            elif(self.args.class_type == 'multi'):
                average = "macro"
            precision = precision_score(targets, predictions, average=average) 
            recall = recall_score(targets, predictions, average=average)
            f1 = f1_score(targets, predictions, average=average)
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            command = "echo 'MACRO RESULT: Accuracy: " + str(accuracy) + "' + 'Precision: " + str(precision) + "' + 'Recall: " + str(recall) + "' + 'F1 Score: " + str(f1) + "'"
            subprocess.call(command, shell=True)
            precision = precision_score(targets, predictions, average="weighted") 
            recall = recall_score(targets, predictions, average="weighted")
            f1 = f1_score(targets, predictions, average="weighted")
            avg_precision += precision
            avg_recall += recall 
            avg_f1 += f1
            count += 1
            command = "echo 'WEIGHTED RESULT: Accuracy: " + str(accuracy) + "' + 'Precision: " + str(precision) + "' + 'Recall: " + str(recall) + "' + 'F1 Score: " + str(f1) + "'\n"
            subprocess.call(command, shell=True)
            confusion_matrix_result = confusion_matrix(targets, predictions)
            print("Confusion Matrix: ", confusion_matrix_result)
            command = "echo 'Confusion Matrix: " + str(confusion_matrix_result) + "'"
            subprocess.call(command, shell=True)
        print("avg precision: ", avg_precision / count)
        print("avg recall: ", avg_recall / count)
        print("avg f1: ", avg_f1 / count)

    
    def test_accuracy_with_weight(self):
        command = "echo 'Testing accuracy with weight!'"
        subprocess.call(command, shell=True)
        test_model = self.Model.build_model(class_type=self.args.class_type)
        optimizer = optim.Adam(test_model.parameters(), lr=self.lr, weight_decay=0.0001)
        print("Testing accuracy with weight!")
        coreset = self.final_coreset
        print("len coreset: ", len(coreset))
        coreset_loader = DataLoader(coreset, batch_size=128, shuffle=False, drop_last=True)
        weights = self.final_weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) * len(coreset)
        weights = torch.from_numpy(weights).float().to(self.device)
        
        # weights = weights / torch.sum(weights)

        # Training loop
        test_model.train()
        test_model = test_model.to(device=self.device)
        num_epochs = 100
        for epoch in range(num_epochs):
            for batch,(input, target, idx) in enumerate(coreset_loader):
                input = input.to(device=self.device, dtype=self.dtype)
                target = target.to(device=self.device, dtype=self.dtype).squeeze().long()
                batch_weight = weights[batch * 128: (batch + 1) * 128]
                optimizer.zero_grad()
                output, _ = test_model(input)
                loss = self.criterion(output,target)
                loss = (loss * batch_weight).mean()
                loss.backward()
                optimizer.step()  
        

        directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_apigraph_drebin'
        test_files = [
            f'{directory}/2013-07_selected.npz',
            f'{directory}/2013-08_selected.npz',
            f'{directory}/2013-09_selected.npz',
            f'{directory}/2013-10_selected.npz',
            f'{directory}/2013-11_selected.npz',
            f'{directory}/2013-12_selected.npz'
        ]
        test_files.extend(glob.glob(f'{directory}/2014*.npz'))
        test_files.extend(glob.glob(f'{directory}/2015*.npz'))
        test_files.extend(glob.glob(f'{directory}/2016*.npz'))
        test_files.extend(glob.glob(f'{directory}/2017*.npz'))
        test_files.extend(glob.glob(f'{directory}/2018*.npz'))

        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        count = 0

        for files in test_files:
            unlabel_loader = self.load_test_data(files)
            test_model.eval()
            predictions = []
            targets = []
            # Disable gradient calculations
            with torch.no_grad():
                for batch, (input, target, idx) in enumerate(unlabel_loader):
                    input = input.to(self.device, dtype=self.dtype)
                    target = target.to(self.device, dtype=self.dtype).squeeze().long()
                    output, _ = test_model(input)
                    probabilities = F.softmax(output, dim=1)
                    _, pseudo_label = torch.max(probabilities, dim=1)
                    predictions.extend(pseudo_label.cpu().numpy())
                    targets.extend(target.cpu().numpy())
            predictions = np.array(predictions)
            targets = np.array(targets)
            accuracy = accuracy_score(targets, predictions)
            if(self.args.class_type == 'binary'):
                average = "binary"
            elif(self.args.class_type == 'multi'):
                average = "macro"
            precision = precision_score(targets, predictions, average=average) 
            recall = recall_score(targets, predictions, average=average)
            f1 = f1_score(targets, predictions, average=average)
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            command = "echo 'MACRO RESULT: Accuracy: " + str(accuracy) + "' + 'Precision: " + str(precision) + "' + 'Recall: " + str(recall) + "' + 'F1 Score: " + str(f1) + "'"
            subprocess.call(command, shell=True)
            precision = precision_score(targets, predictions, average="weighted") 
            recall = recall_score(targets, predictions, average="weighted")
            f1 = f1_score(targets, predictions, average="weighted")
            avg_precision += precision
            avg_recall += recall 
            avg_f1 += f1
            count += 1
            command = "echo 'WEIGHTED RESULT: Accuracy: " + str(accuracy) + "' + 'Precision: " + str(precision) + "' + 'Recall: " + str(recall) + "' + 'F1 Score: " + str(f1) + "'\n"
            subprocess.call(command, shell=True)
            confusion_matrix_result = confusion_matrix(targets, predictions)
            print("Confusion Matrix: ", confusion_matrix_result)
            command = "echo 'Confusion Matrix: " + str(confusion_matrix_result) + "'"
            subprocess.call(command, shell=True)
        print("avg precision: ", avg_precision / count)
        print("avg recall: ", avg_recall / count)
        print("avg f1: ", avg_f1 / count)




    def random_test(self):     
        directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_apigraph_drebin'
        test_files = [
            f'{directory}/2013-07_selected.npz',
            f'{directory}/2013-08_selected.npz',
            f'{directory}/2013-09_selected.npz',
            f'{directory}/2013-10_selected.npz',
            f'{directory}/2013-11_selected.npz',
            f'{directory}/2013-12_selected.npz'
        ]
        test_files.extend(glob.glob(f'{directory}/2014*.npz'))
        test_files.extend(glob.glob(f'{directory}/2015*.npz'))
        test_files.extend(glob.glob(f'{directory}/2016*.npz'))
        test_files.extend(glob.glob(f'{directory}/2017*.npz'))
        test_files.extend(glob.glob(f'{directory}/2018*.npz'))

        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        count = 0

        for files in test_files:
            unlabel_loader = self.load_test_data(files)
            self.model.eval()
            predictions = []
            targets = []
            # Disable gradient calculations
            with torch.no_grad():
                for batch, (input, target, idx) in enumerate(unlabel_loader):
                    input = input.to(self.device, dtype=self.dtype)
                    target = target.to(self.device, dtype=self.dtype).squeeze().long()
                    output, _ = self.model(input)
                    probabilities = F.softmax(output, dim=1)
                    _, pseudo_label = torch.max(probabilities, dim=1)
                    predictions.extend(pseudo_label.cpu().numpy())
                    targets.extend(target.cpu().numpy())
            predictions = np.array(predictions)
            targets = np.array(targets)
            accuracy = accuracy_score(targets, predictions)
            if(self.args.class_type == 'binary'):
                average = "binary"
            elif(self.args.class_type == 'multi'):
                average = "macro"
            precision = precision_score(targets, predictions, average=average) 
            recall = recall_score(targets, predictions, average=average)
            f1 = f1_score(targets, predictions, average=average)
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            command = "echo 'MACRO RESULT: Accuracy: " + str(accuracy) + "' + 'Precision: " + str(precision) + "' + 'Recall: " + str(recall) + "' + 'F1 Score: " + str(f1) + "'"
            subprocess.call(command, shell=True)
            precision = precision_score(targets, predictions, average="weighted") 
            recall = recall_score(targets, predictions, average="weighted")
            f1 = f1_score(targets, predictions, average="weighted")
            avg_precision += precision
            avg_recall += recall 
            avg_f1 += f1
            count += 1
            command = "echo 'WEIGHTED RESULT: Accuracy: " + str(accuracy) + "' + 'Precision: " + str(precision) + "' + 'Recall: " + str(recall) + "' + 'F1 Score: " + str(f1) + "'\n"
            subprocess.call(command, shell=True)
            confusion_matrix_result = confusion_matrix(targets, predictions)
            print("Confusion Matrix: ", confusion_matrix_result)
            command = "echo 'Confusion Matrix: " + str(confusion_matrix_result) + "'"
            subprocess.call(command, shell=True)
        print("avg precision: ", avg_precision / count)
        print("avg recall: ", avg_recall / count)
        print("avg f1: ", avg_f1 / count)
    
    
caller = main_trainer(args=args)
caller.main_train(100)
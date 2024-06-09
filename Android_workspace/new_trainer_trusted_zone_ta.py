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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

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
        self.batch_size = 1024 # Batch size
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
        # self.rs_rate = 0.001 # In IID, the rate of random sampling that is used to train the initial model
        self.budget_ratio = self.args.budget # Budget
        self.budget = self.args.budget
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
        
        self.directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_apigraph_drebin'
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

        self.steps_per_epoch = 10

        self.eigenv = np.load(f'{self.directory}/eigvecs_d_2.npy') # Load eigenvectors
        self.eigenv = torch.tensor(self.eigenv, dtype=self.dtype, device=self.device)

        self.f1 = 0
        self.FNR = 0
        self.FPR = 0

    
    def load_file(self, file):
        data = np.load(file)
        x_train = data['X_train']
        y_train = data['y_train']

        y_train_binary = np.where(y_train == 0, 0, 1) # Convert to binary

        y_mal_family = data['y_mal_family']
        return x_train, y_train_binary, y_mal_family
    
    def get_loader(self, file):
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

        x_data = torch.from_numpy(x_data.values)
        y_data = torch.from_numpy(y_data.values)
        full_dataset = TensorDataset(x_data, y_data)
        x_data = full_dataset.tensors[0].numpy()
        x_data = torch.from_numpy(x_data).unsqueeze(1)
        full_dataset = TensorDataset(x_data, full_dataset.tensors[1])
        loader = DataLoader(indexed_Dataset.IndexedDataset(full_dataset), batch_size=self.batch_size, shuffle=True, drop_last=False)
        return loader 

    
 
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
                continuous_state = uncertainty_similarity.continuous_states(self.eigenv, self.label_loader, self.unlabel_loader, self.model, self.device, self.dtype, alpha=self.alpha, sigma=self.sigma)
                self.alpha = min(self.alpha + 0.01, self.alpha_max)
                continuous_state = continuous_state[:, None]
                command = "echo 'Continuous state shape: " + str(len(continuous_state)) + "'"
                subprocess.call(command, shell=True)


                self.gradients = self.Model_trainer.gradient_train(training_step, self.model, self.unlabel_loader, self.pseudo_labels, self.device, self.dtype, batch_size=self.batch_size, criterion=self.criterion)
                command = "echo 'Gradients shape: " + str(len(self.gradients)) + "'"
                subprocess.call(command, shell=True)

                self.gradients = self.gradients * continuous_state
                # ---------------------print end---------------------
                
                
                # ---------------------print begin---------------------
                command = "echo 'select_subset at step: " + str(training_step) + "'"
                subprocess.call(command, shell=True)
                self.select_subset(training_step)
                command = "echo 'select_subset complete!'"
                subprocess.call(command, shell=True)
                # ---------------------print end---------------------
                
                # for weight in self.weights:
                #     self.final_weights.append(weight)
                
                self.update_train_loader_and_weights(training_step)

                # Update the train loader and weights
                # if self.final_coreset is None:
                #     self.final_coreset = Subset(self.unlabel_loader.dataset, self.coreset_index)
                # else:
                    # if(len(self.final_coreset) > self.budget):
                    #     print("Budget reached!")
                    #     self.stop = 1
                    #     break
                    # self.final_coreset = ConcatDataset([self.final_coreset, Subset(self.unlabel_loader.dataset, self.coreset_index)])
                # ---------------------print begin---------------------
                # command = "echo 'Final coreset length at step: " + str(training_step) + " is " + str(len(self.final_coreset)) + "'"
                # subprocess.call(command, shell=True)
                # ---------------------print end---------------------

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
    
    def main_train(self):
        epoch = len(self.test_files)
        self.label_loader = self.get_loader(self.training_file[0])

        command = "echo 'Initial training...'"
        subprocess.call(command, shell=True)
        self.Model_trainer.initial_train(self.label_loader, self.model, self.device, self.dtype, criterion=self.criterion, learning_rate=self.lr)
        command = "echo 'Initial training complete!'"
        subprocess.call(command, shell=True)


        for e in range(epoch-1):
            command = "echo 'Training file: " + str(self.test_files[e]) + "'"
            subprocess.call(command, shell=True)
            self.unlabel_loader = self.get_loader(self.test_files[e])
            
            command = "echo 'Acquiring pseudo labels...'"
            subprocess.call(command, shell=True)
            self.pseudo_labels = self.Model_trainer.psuedo_labeling(self.model, self.device, self.dtype, loader=self.unlabel_loader)
            command = "echo 'Pseudo labels acquired!'"
            subprocess.call(command, shell=True)

            for i in range(4):
                self.train_epoch(i)
            command = "echo 'Testing file: " + str(self.test_files[e+1]) + "'"
            subprocess.call(command, shell=True)
            self.test_accuracy(self.test_files[e+1])
        
        self.f1 = self.f1 / epoch
        self.FNR = self.FNR / epoch
        self.FPR = self.FPR / epoch
        command = "echo 'Average f1 score: " + str(self.f1) + "'"
        subprocess.call(command, shell=True)
        command = "echo 'Average FNR (False Negative Rate): " + str(self.FNR) + "' + 'Average FPR (False Positive Rate): " + str(self.FPR) + "'"
        subprocess.call(command, shell=True)
        print("Average f1 score: ", self.f1)
        print("Average FNR (False Negative Rate): ", self.FNR)
        print("Average FPR (False Positive Rate): ", self.FPR)
        
            

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
            indices_per_class = np.random.choice(class_indices, size=int(np.ceil(0.1 * len(class_indices))), replace=False)
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
            sub_coreset_index, sub_weights= facility_Update.get_orders_and_weights(5, gradient_data, "euclidean", y=fl_labels.cpu().numpy(), equal_num=False, mode="sparse", num_n=64)
            sub_coreset_index = subset[sub_coreset_index] # Get the indices of the coreset
            self.coreset_index.extend(sub_coreset_index.tolist()) # Add the coreset to the coreset list
            self.weights.extend(sub_weights.tolist()) # Add the weights to the weights list
            self.subsets.extend(subset)
            subset_count += 1
        print("Greedy FL Complete at step: ", training_step)

    def load_test_data(self, file):
        x_train = []
        y_train = []
        y_mal_family = []
        command = "echo 'Loading test file: " + str(file) + "'"
        subprocess.call(command, shell=True)
        x, y, y_mal = self.load_file(os.path.join(self.directory, file))
        x_train.append(x)
        y_train.append(y)
        y_mal_family.append(y_mal)
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        x_data = pd.DataFrame(x_train).astype(float)
        y_data = pd.DataFrame(y_train).astype(float)
        x_data = torch.from_numpy(x_data.values)
        y_data = torch.from_numpy(y_data.values)
        full_dataset = TensorDataset(x_data, y_data)
        x_data = full_dataset.tensors[0].numpy()
        x_data = torch.from_numpy(x_data).unsqueeze(1)
        full_dataset = TensorDataset(x_data, full_dataset.tensors[1])
        unlabel_loader = DataLoader(indexed_Dataset.IndexedDataset(full_dataset), batch_size=self.batch_size, shuffle=False, drop_last=True)
        return unlabel_loader
    
    def test_accuracy(self, file):
        unlabel_loader = self.load_test_data(file)
        test_model = self.model
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

        confusion_matrix_result = confusion_matrix(targets, predictions)
        print("Confusion Matrix: ", confusion_matrix_result)
        command = "echo 'Confusion Matrix: " + str(confusion_matrix_result) + "'"
        subprocess.call(command, shell=True)
        TN, FP, FN, TP = confusion_matrix_result.ravel()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        FNR = FN / (FN + TP)
        FPR = FP / (FP + TN)
        print(f"f1 score: {f1:.2f}")
        print(f"FNR (False Negative Rate): {FNR:.2f}")
        print(f"FPR (False Positive Rate): {FPR:.2f}")
        command = "echo 'f1 score: " + str(f1) + "'"
        subprocess.call(command, shell=True)
        command = "echo 'FNR (False Negative Rate): " + str(FNR) + "' + 'FPR (False Positive Rate): " + str(FPR) + "'"
        subprocess.call(command, shell=True)
        self.f1 += f1
        self.FNR += FNR
        self.FPR += FPR
    
    

caller = main_trainer(args=args)
caller.main_train()
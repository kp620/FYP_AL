# --------------------------------
# Import area
import Test_cuda, Train_model_trainer_multiclass, Data_wrapper_multiclass, Approx_optimizer, restnet_1d_multiclass, Facility_Update, IndexedDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn.functional as F

import subprocess

# --------------------------------
# Main processing logic

class new_trainer():
    def __init__(self):
        # Device & Data type
        self.device, self.dtype = Test_cuda.check_device() # Check if cuda is available

        # Model & Parameters
        self.label_model = restnet_1d_multiclass.build_model() # Model used to train the data(M_1)
        self.train_model = restnet_1d_multiclass.build_model() # Model used to train the data(M_0)
        self.batch_size = 128 # Batch size
        self.lr = 0.00001 # Learning rate
        self.optimizer = optim.Adam(self.train_model.parameters(), lr=self.lr) # Optimizer
        self.criterion = nn.CrossEntropyLoss() # Loss function

        # Counter
        self.steps_per_epoch = int
        self.reset_step = int
        self.train_iter = iter
        
        # Variables
        self.rs_rate = 0.05
        self.gradients = [] # Gradients of the unlabel set
        self.pseudo_labels = [] # Pseudo labels of the unlabel set
        self.unlabel_loader = DataLoader # Unlabelled dataset
        self.train_loader = DataLoader # Training dataset
        self.approx_loader = DataLoader # Approximation dataset
        self.coreset_loader = DataLoader # Coreset dataset
        self.check_thresh_factor = 0.1
        self.gradient_approx_optimizer = Approx_optimizer.Adahessian(self.train_model.parameters())
        self.delta = 0 
        self.gf = 0 
        self.ggf = 0
        self.ggf_moment = 0
        self.start_loss = 0
        self.subsets = []
        self.coreset_index = []
        self.weights = []
        self.train_weights = []
        self.final_coreset = None
        self.final_weights = []
    
    # Given the initial dataset, select a subset of the dataset to train the initial model M_0
    def initial_training(self):
        # Load training data, acquire label and unlabel set using rs_rate / us_rate
        ini_train_loader, self.unlabel_loader = Data_wrapper_multiclass.process_rs(batch_size=self.batch_size, rs_rate=self.rs_rate)
        # ini_train_loader, self.unlabel_loader = Data_wrapper.process_us(self.train_model, self.device, self.dtype, self.batch_size, self.rs_rate) 

        # Train the initial model over the label set
        Train_model_trainer_multiclass.initial_train(ini_train_loader, self.train_model, self.device, self.dtype, criterion=self.criterion, learning_rate=self.lr)
        # Acquire pseudo labels of the unlabel set
        self.pseudo_labels = Train_model_trainer_multiclass.psuedo_labeling(self.train_model, self.device, self.dtype, loader = self.unlabel_loader)

 
    def train_epoch(self, epoch):
        self.steps_per_epoch = np.ceil(int(len(self.unlabel_loader) * 0.1) / self.batch_size).astype(int)
        print("steps per epoch: ", self.steps_per_epoch)
        self.reset_step = self.steps_per_epoch
        self.train_model.train()

        for training_step in range(self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)):
            if((training_step > self.reset_step) and ((training_step - self.reset_step) % 3 == 0)): 
                self.check_approx_error(training_step)
            
            if training_step == self.reset_step or training_step == 0:
                self.gradients = Train_model_trainer_multiclass.gradient_train(self.train_model, self.unlabel_loader, self.pseudo_labels, self.device, self.dtype, batch_size = self.batch_size, criterion = self.criterion)
                print("len gradients: ", len(self.gradients))
                self.select_subset()
                self.update_train_loader_and_weights()

                if self.final_coreset is None:
                    # Directly assign the first dataset
                    self.final_coreset = Subset(self.unlabel_loader.dataset, self.coreset_index)
                else:
                    # Concatenate additional datasets
                    self.final_coreset = ConcatDataset([self.final_coreset, Subset(self.unlabel_loader.dataset, self.coreset_index)])
                for weight in self.weights:
                    self.final_weights.append(weight)


                self.train_loader = self.coreset_loader
                self.train_iter = iter(self.train_loader)
                self.get_quadratic_approximation()

            try: 
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)

            data, target, data_idx = batch
            data, target = data.to(self.device, dtype=self.dtype), target.to(self.device, dtype=self.dtype).squeeze().long()
            self.forward_and_backward(data, target, data_idx)
    
    def main_train(self, epoch):
        self.initial_training() # 在initial training就update delta?

        for e in range(epoch):
            print("Epoch: ", e)
            command = ["echo", f"Epoch: {e}"]
            subprocess.run(command)

            self.train_epoch(e)

            if e % 3 == 1:
                accuracy_without_weight = self.test_accuracy_without_weight()
                command = ["echo", f"Accuracy without weight: {accuracy_without_weight}"]
                subprocess.run(command)

                accuracy_with_weight = self.test_accuracy_with_weight()
                command = ["echo", f"Accuracy with weight: {accuracy_with_weight}"]
                subprocess.run(command)
        
        print("Training complete!")
        self.test_accuracy_without_weight()
        self.test_accuracy_with_weight()


# --------------------------------
# Auxiliary functions     
    def forward_and_backward(self, data, target, data_idx):
        self.optimizer.zero_grad()

        # train model with the current batch and record forward and backward time
        output, _ = self.train_model(data)

        loss = self.criterion(output, target)
        loss = (loss * self.train_weights[data_idx]).mean()

        # compute the parameter change delta
        self.train_model.zero_grad()
        # approximate with hessian diagonal
        loss.backward(create_graph=True)
        gf_current, _, _ = self.gradient_approx_optimizer.step(momentum=False)                   
        self.delta -= self.lr * gf_current

        loss.backward()
        self.optimizer.step()


    def check_approx_error(self, training_step):
        # calculate true loss    
        true_loss = 0 
        count = 0 
        subset_loader = DataLoader(Subset(self.unlabel_loader.dataset, self.subsets), batch_size=self.batch_size, shuffle=False, drop_last=True)
        print("len subset_loader: ", len(Subset(self.unlabel_loader.dataset, self.subsets)))

        self.train_model.eval()
        for approx_batch, (input, target, idx) in enumerate(subset_loader):
            self.optimizer.zero_grad()
            input = input.to(self.device, dtype=self.dtype)
            pseudo_y = self.pseudo_labels[idx].to(self.device, dtype=self.dtype).squeeze().long()

            output,_ = self.train_model(input)

            loss = self.criterion(output, pseudo_y)
            
            loss.backward()
            self.optimizer.step()
            true_loss += loss.item() * input.size(0)
            count += input.size(0)

        self.train_model.train()
        true_loss = true_loss / count

        # self.train_model.train()
        approx_loss = torch.matmul(self.delta, self.gf) + self.start_loss
        approx_loss += 1/2 * torch.matmul(self.delta * self.ggf, self.delta)

        loss_diff = abs(true_loss - approx_loss.item())
        print("true loss: ", true_loss)
        print("approx loss: ", approx_loss)
        print("diff: ", loss_diff)

        #------------TODO--------------
        thresh = self.check_thresh_factor * true_loss                          
        #------------TODO--------------

        if loss_diff > thresh:
            self.reset_step = training_step
            all_indices = set(range(len(self.unlabel_loader.dataset)))
            coreset_indices = set(self.coreset_index)
            remaining_indices = list(all_indices - coreset_indices)
            unlabel_set = self.unlabel_loader.dataset.dataset
            unlabel_set = Subset(unlabel_set, remaining_indices)
            self.unlabel_loader = DataLoader(IndexedDataset.IndexedDataset(unlabel_set), batch_size=self.batch_size, shuffle=False, drop_last=True)
            self.pseudo_labels = Train_model_trainer_multiclass.psuedo_labeling(self.train_model, self.device, self.dtype, loader = self.unlabel_loader)
            
    
    def get_quadratic_approximation(self):
        # second-order approximation with coreset
        self.approx_loader = DataLoader(
            Subset(self.unlabel_loader.dataset, indices=self.coreset_index),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        print("len approx_loader: ", len(Subset(self.unlabel_loader.dataset, indices=self.coreset_index)))

        self.start_loss = 0 
        count = 0 
        for approx_batch, (input, target, idx) in enumerate (self.approx_loader):
            input = input.to(self.device, dtype=self.dtype)
            # target = target.to(self.device, dtype=self.dtype).squeeze().long()
            pseudo_y = self.pseudo_labels[idx].to(self.device, dtype=self.dtype).squeeze().long()
            output, _ = self.train_model(input)

            # for coreset
            # loss = self.criterion(output, target)
            loss = self.criterion(output, pseudo_y)
            batch_weight = self.train_weights[idx.long()]
            loss = (loss * batch_weight).mean()

            self.train_model.zero_grad()
            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(momentum=True)
            if approx_batch == 0:
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
        self.delta = 0

        print("Quadratic approximation complete!")

    def update_train_loader_and_weights(self):
        self.coreset_loader = DataLoader(
            Subset(self.unlabel_loader.dataset, indices = self.coreset_index),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.train_weights = np.zeros(len(self.unlabel_loader.dataset))
        self.weights = self.weights / np.sum(self.weights) * len(self.coreset_index)
        self.train_weights[self.coreset_index] = self.weights
        self.train_weights = torch.from_numpy(self.train_weights).float().to(self.device)

    def select_random_set(self):
        train_indices = np.arange(len(self.gradients))
        indices = []
        for c in np.unique(self.pseudo_labels):
            class_indices = np.intersect1d(np.where(self.pseudo_labels == c)[0], train_indices)
            indices_per_class = np.random.choice(class_indices, size=int(np.ceil(0.001 * len(class_indices))), replace=False)
            indices.append(indices_per_class)
        indices = np.concatenate(indices).astype(int)
        return indices

    def select_subset(self):
        subsets = []
        self.coreset_index = []
        self.subsets = []
        self.weights = []
        for _ in range(5):
            # get a random subset of the data
            random_subset = self.select_random_set()
            subsets.append(random_subset)

        print("Greedy FL Start!")
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
                    
            # Facility location function
            fl_labels = self.pseudo_labels[subset] - torch.min(self.pseudo_labels[subset])

            sub_coreset_index, sub_weights= Facility_Update.get_orders_and_weights(
                128,
                gradient_data,
                "euclidean",
                y=fl_labels.cpu().numpy(),
                equal_num=False,
                mode="sparse",
                num_n=128,
            )

            sub_coreset_index = subset[sub_coreset_index] # Get the indices of the coreset
            self.coreset_index.extend(sub_coreset_index.tolist()) # Add the coreset to the coreset list
            self.weights.extend(sub_weights.tolist()) # Add the weights to the weights list
            self.subsets.extend(subset)
            subset_count += 1

    def test_accuracy_without_weight(self):
        test_model = restnet_1d_multiclass.build_model()
        optimizer = optim.Adam(test_model.parameters(), lr=self.lr)
        print("Testing accuracy without weight!")
        unlabel_loader = self.unlabel_loader
        coreset = self.final_coreset
        print("len coreset: ", len(coreset))
        command = ["echo", f"Length of coreset: {len(coreset)}"]
        subprocess.run(command)

        coreset_loader = DataLoader(coreset, batch_size=self.batch_size, shuffle=True)
        test_model.train()
        test_model = test_model.to(device=self.device)

        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            for t,(x,y,idx) in enumerate(coreset_loader):
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype).squeeze().long()
                optimizer.zero_grad()
                output, _ = test_model(x)
                loss = self.criterion(output,y)
                loss.backward()
                optimizer.step()  
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        test_model.eval()
        test_model = test_model.to(device=self.device)
        correct_predictions = 0
        total_predictions = 0
        # Disable gradient calculations
        with torch.no_grad():
            for t, (inputs, targets, idx) in enumerate(unlabel_loader):
                inputs = inputs.to(self.device, dtype=self.dtype)
                targets = targets.to(self.device, dtype=self.dtype).squeeze().long()
                # Forward pass to get outputs
                output, _ = test_model(inputs)
                # Calculate the loss
                loss = self.criterion(output, targets)
                probabilities = F.softmax(output, dim=1)
                _, pseudo_label = torch.max(probabilities, dim=1)
                # Count correct predictions
                correct_predictions += torch.sum(pseudo_label == targets.data).item()
                total_predictions += targets.size(0)
        # Calculate average loss and accuracy
        test_accuracy = correct_predictions / total_predictions
        print("correct predictions without weight: ", correct_predictions)
        print("total_predictions without weight: ", total_predictions)
        print(f'Test Accuracy without weight: {test_accuracy:.2f}')
        return test_accuracy
    
    def test_accuracy_with_weight(self):
        test_model = restnet_1d_multiclass.build_model()
        print("Testing accuracy with weight!")
        unlabel_loader = self.unlabel_loader
        coreset = self.final_coreset
        print("len coreset: ", len(coreset))
        command = ["echo", f"Length of coreset: {len(coreset)}"]
        subprocess.run(command)
        
        weights = self.final_weights

        coreset_loader = DataLoader(coreset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_model.train()
        test_model = test_model.to(device=self.device)
        optimizer = optim.Adam(test_model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        weights = np.array(weights)
        weights = torch.from_numpy(weights).float().to(self.device)
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            for t,(x,y,idx) in enumerate(coreset_loader):
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype).squeeze().long()
                batch_weight = weights[t*self.batch_size:(t+1)*self.batch_size]
                optimizer.zero_grad()
                output, _ = test_model(x)
                loss = criterion(output,y)
                loss = (loss * batch_weight).mean()
                loss.backward()
                optimizer.step()  
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        test_model.eval()
        test_model = test_model.to(device=self.device)
        correct_predictions = 0
        total_predictions = 0
        # Disable gradient calculations
        with torch.no_grad():
            for t, (inputs, targets, _) in enumerate(unlabel_loader):
                inputs = inputs.to(self.device, dtype=self.dtype)
                targets = targets.to(self.device, dtype=self.dtype).squeeze().long()
                # Forward pass to get outputs
                output, _ = test_model(inputs)
                # Calculate the loss
                loss = criterion(output, targets)
                probabilities = F.softmax(output, dim=1)
                _, pseudo_label = torch.max(probabilities, dim=1)
                # Count correct predictions
                correct_predictions += torch.sum(pseudo_label == targets.data).item()
                total_predictions += targets.size(0)
        # Calculate average loss and accuracy
        test_accuracy = correct_predictions / total_predictions
        print("correct predictions with weight: ", correct_predictions)
        print("total_predictions with weight: ", total_predictions)
        print(f'Test Accuracy with weight: {test_accuracy:.2f}')
        return test_accuracy
    
caller = new_trainer()
caller.main_train(10)
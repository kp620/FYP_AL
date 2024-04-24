# --------------------------------
# Import area
import test_Cuda, model_Trainer, data_Wrapper, approx_Optimizer, restnet_1d, facility_Update, indexed_Dataset, data_preprocess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn.functional as F
import subprocess
import pickle
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --------------------------------
# Argument parser
# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument("--operation_type",  choices=['iid', 'time-aware'], help='Specify operation type: "iid" or "time-aware"')
parser.add_argument("--class_type", choices=['binary', 'multi'], help='Specify class type: "binary" or "multi"')
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
        self.unlabel_loader = DataLoader # Unlabelled dataset
        self.train_loader = DataLoader # Training dataset
        self.approx_loader = DataLoader # Approximation dataset
        self.coreset_loader = DataLoader # Coreset dataset
        self.budget = 0 # Budget
        self.check_thresh_factor = 0.1
        self.delta = 0 
        self.gf = 0 
        self.ggf = 0
        self.ggf_moment = 0
        self.start_loss = 0
        self.rs_rate = 0.005 # In IID, the rate of random sampling that is used to train the initial model
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
    
    
    # Given the initial dataset, select a subset of the dataset to train the initial model M_0
    def initial_training(self):
        # Load training data, acquire label and unlabel set using rs_rate / us_rate
        if(self.args.operation_type == 'iid'):
            # ini_train_loader, self.unlabel_loader, self.budget = data_Wrapper.process_rs(batch_size=self.batch_size, rs_rate=self.rs_rate, class_type=self.args.class_type, budget=0.03) 
            ini_train_loader, self.unlabel_loader, self.budget = data_preprocess.main(batch_size=self.batch_size, rs_rate=self.rs_rate, class_type=self.args.class_type, budget=0.015)
        # Train the initial model over the label set
        self.Model_trainer.initial_train(ini_train_loader, self.model, self.device, self.dtype, criterion=self.criterion, learning_rate=self.lr)
        # Acquire pseudo labels of the unlabel set
        self.pseudo_labels = self.Model_trainer.psuedo_labeling(self.model, self.device, self.dtype, loader=self.unlabel_loader)
        self.steps_per_epoch = np.ceil(int(len(self.unlabel_loader.dataset.dataset) * 0.015) / self.batch_size).astype(int)
        print("Steps per epoch: ", self.steps_per_epoch)

 
    def train_epoch(self, epoch):
        print("Training epoch: ", epoch)
        # self.steps_per_epoch = np.ceil(int(len(self.unlabel_loader.dataset.dataset) * 0.1) / self.batch_size).astype(int)
        # print("Steps per epoch: ", self.steps_per_epoch)
        self.reset_step = self.steps_per_epoch
        # Training loop
        self.model.train()
        for training_step in range(self.steps_per_epoch * epoch, self.steps_per_epoch * (epoch + 1)):
            # Check the approximation error
            if((training_step > self.reset_step) and ((training_step - self.reset_step) % 2 == 0)): 
                print("Checking approx error at step: ", training_step)
                self.check_approx_error(training_step)
            # Update the coreset
            if training_step == self.reset_step or training_step == 0:
                print("Updating coreset at step: ", training_step)
                self.gradients = self.Model_trainer.gradient_train(training_step, self.model, self.unlabel_loader, self.pseudo_labels, self.device, self.dtype, batch_size=self.batch_size, criterion=self.criterion)
                self.select_subset(training_step)
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

            data, target, idx = batch
            data, target = data.to(self.device, dtype=self.dtype), target.to(self.device, dtype=self.dtype).squeeze().long()
            self.forward_and_backward(data, target, idx)
    
    def main_train(self, epoch):
        self.initial_training() # 在initial training就update delta?
        print("Main training start!")

        for e in range(epoch):
            command = "echo 'Epoch: " + str(e) + "'"
            subprocess.call(command, shell=True)
            self.train_epoch(e)

            self.test_accuracy_without_weight()
            self.test_accuracy_with_weight()

            if self.stop == 1:
                break

        print("Main training complete!")
        self.test_accuracy_without_weight()
        self.test_accuracy_with_weight()




# --------------------------------
# Auxiliary functions     
    def forward_and_backward(self, data, target, idx):
        self.optimizer.zero_grad()
        output, _ = self.model(data)
        loss = self.criterion(output, target)
        loss = (loss * self.train_weights[idx]).mean()
        self.model.zero_grad()
        loss.backward(create_graph=True)
        gf_current, _, _ = self.gradient_approx_optimizer.step(momentum=False)         
        self.delta -= self.lr * gf_current
        loss.backward()
        self.optimizer.step()


    def check_approx_error(self, training_step):
        # calculate true loss    
        true_loss = 0 
        count = 0 
        subset_loader = DataLoader(Subset(self.unlabel_loader.dataset, self.coreset_index), batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.model.eval()
        for batch, (data, target, idx) in enumerate(subset_loader):
            self.optimizer.zero_grad()
            data = data.to(self.device, dtype=self.dtype)
            # pseudo_y = self.pseudo_labels[idx].to(self.device, dtype=self.dtype).squeeze().long()
            pseudo_y = target.to(self.device, dtype=self.dtype).squeeze().long()
            output,_ = self.model(data)
            loss = self.criterion(output, pseudo_y)
            loss.backward()
            self.optimizer.step()
            true_loss += loss.item() * data.size(0)
            count += data.size(0)

        self.model.train()
        true_loss = true_loss / count
        approx_loss = torch.matmul(self.delta, self.gf) + self.start_loss
        approx_loss += 1/2 * torch.matmul(self.delta * self.ggf, self.delta)
        loss_diff = abs(true_loss - approx_loss.item())
        print("true loss: ", true_loss)
        print("approx loss: ", approx_loss)
        print("diff: ", loss_diff)
        # command = "echo 'True loss: " + str(true_loss) + "' + 'Approx loss: " + str(approx_loss) + "' + 'Diff: " + str(loss_diff) + "'"
        # subprocess.call(command, shell=True)
        # command = "echo 'Start loss: " + str(self.start_loss) + "'"
        # subprocess.call(command, shell=True)

        thresh = self.check_thresh_factor * true_loss                          
        if loss_diff > thresh:
            print("Loss diff is larger than threshold, updating coreset at step: ", training_step)
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
            target = target.to(self.device, dtype=self.dtype).squeeze().long()
            output, _ = self.model(input)
            loss = self.criterion(output, target)
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
        self.delta = 0
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
        for iteration in range(5):
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
            sub_coreset_index, sub_weights= facility_Update.get_orders_and_weights(64, gradient_data, "euclidean", y=fl_labels.cpu().numpy(), equal_num=False, mode="sparse", num_n=128)
            sub_coreset_index = subset[sub_coreset_index] # Get the indices of the coreset
            self.coreset_index.extend(sub_coreset_index.tolist()) # Add the coreset to the coreset list
            self.weights.extend(sub_weights.tolist()) # Add the weights to the weights list
            self.subsets.extend(subset)
            subset_count += 1
        print("Greedy FL Complete at step: ", training_step)

    def test_accuracy_without_weight(self):
        command = "echo 'Testing accuracy without weight!'"
        subprocess.call(command, shell=True)
        test_model = self.Model.build_model(class_type=self.args.class_type)
        optimizer = optim.Adam(test_model.parameters(), lr=self.lr)
        print("Testing accuracy without weight!")
        unlabel_loader = self.unlabel_loader
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
        command = "echo 'WEIGHTED RESULT: Accuracy: " + str(accuracy) + "' + 'Precision: " + str(precision) + "' + 'Recall: " + str(recall) + "' + 'F1 Score: " + str(f1) + "'\n"
        subprocess.call(command, shell=True)
        confusion_matrix_result = confusion_matrix(targets, predictions)
        print("Confusion Matrix: ", confusion_matrix_result)
        command = "echo 'Confusion Matrix: " + str(confusion_matrix_result) + "'"
        subprocess.call(command, shell=True)
    
    def test_accuracy_with_weight(self):
        command = "echo 'Testing accuracy with weight!'"
        subprocess.call(command, shell=True)
        test_model = self.Model.build_model(class_type=self.args.class_type)
        optimizer = optim.Adam(test_model.parameters(), lr=self.lr)
        print("Testing accuracy with weight!")
        unlabel_loader = self.unlabel_loader
        coreset = self.final_coreset
        print("len coreset: ", len(coreset))
        coreset_loader = DataLoader(coreset, batch_size=128, shuffle=False, drop_last=True)
        weights = self.final_weights
        weights = np.array(weights)
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
        command = "echo 'WEIGHTED RESULT: Accuracy: " + str(accuracy) + "' + 'Precision: " + str(precision) + "' + 'Recall: " + str(recall) + "' + 'F1 Score: " + str(f1) + "'\n"
        subprocess.call(command, shell=True)
        confusion_matrix_result = confusion_matrix(targets, predictions)
        print("Confusion Matrix: ", confusion_matrix_result)
        command = "echo 'Confusion Matrix: " + str(confusion_matrix_result) + "'"
        subprocess.call(command, shell=True)
    
caller = main_trainer(args=args)
caller.main_train(5)
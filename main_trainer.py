# --------------------------------
# Import area
import Test_cuda, Train_model_trainer, Train_model, Data_wrapper, Gradient_model_trainer, Gradient_model, Facility_Location, Approx_optimizer, restnet_1d, Facility_Update
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.nn.functional as F

# --------------------------------
# Main processing logic

class main_trainer():
    def __init__(self) -> None:
        self.device, self.dtype = Test_cuda.check_device() # Check if cuda is available
        # Model & Parameters
        self.train_model = restnet_1d.build_model() # Model used to train the data(M_0)
        self.batch_size = 1024 # Batch size
        self.lr = 0.00001 # Learning rate
        # Counter
        self.update_coreset = 1 # Decide if coreset needs to be updated
        self.update_times = 0 # Number of times the coreset has been updated
        self.update_times_limit = 5 # Maximum number of times the coreset can be updated (SELF_TUNE!!!)
        # Variables
        self.unlabel_loader = None # Set of unlabelled data
        self.approx_loader = None # Set of data used to approximate the loss
        self.pseudo_labels = [] # Store pseudo labels of unlabel_set
        self.gradients = [] # Store gradients of unlabel_set
        self.coreset = [] # Store coreset
        self.weights = [] # Store weights
        self.rs_rate = 0.05 # Percentage of the dataset to acuquire manual labelling
        self.num_subsets = 2500 # Number of random subsets
        self.subset_size = 0 # Size of each subset
        self.gf = 0 # Store gradient
        self.ggf = 0 # Store hutchinson trace
        self.ggf_moment = 0 # Store hutchinson trace moment
        self.delta = 0 # Store delta
        self.start_loss = 0 # Store start loss
        self.check_thresh_factor = 0.1 # Threshold factor (SELF_TUNE!!!)
        self.subsets = []
        self.gradient_approx_optimizer = Approx_optimizer.Adahessian(self.train_model.parameters())
        self.final_coreset = None
        self.final_weights = []

    def reset(self):
        self.gradients = []
        self.approx_loader = None
        self.coreset = []
        self.weights = []
        self.subsets = []
        self.gf = 0
        self.ggf = 0
        self.ggf_moment = 0
        self.start_loss = 0

    # Given the initial dataset, select a subset of the dataset to train the initial model M_0
    def initial_training(self):
        # Load training data, acquire label and unlabel set using rs_rate / us_rate
        ini_train_loader, unlabel_set = Data_wrapper.process_rs(batch_size=self.batch_size, rs_rate=self.rs_rate)
        # ini_train_loader, unlabel_set = Data_wrapper.process_us(self.train_model, self.device, self.dtype, self.batch_size, self.rs_rate)

        # Update unlabel_loader (Need to update in each iteration)
        self.unlabel_loader =  DataLoader(unlabel_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        # Train the initial model over the label set
        Train_model_trainer.initial_train(ini_train_loader, self.train_model, self.device, self.dtype, criterion=nn.CrossEntropyLoss(), learning_rate=self.lr)
        # Acquire pseudo labels of the unlabel set
        self.pseudo_labels = Train_model_trainer.psuedo_labeling(self.train_model, self.device, self.dtype, loader = self.unlabel_loader)

    def main_loop(self):
        # Get unlabel_loader & pseudo_labels
        self.initial_training()
        # Main loop
        while self.update_times < self.update_times_limit:
            # Update coreset!
            # We initialy set update_coreset to 1, so the first time we enter the loop, we will update coreset
            if self.update_coreset == 1:
                print("Update coreset!")
                # Use the train model to acquire gradients
                self.gradients = Train_model_trainer.gradient_train(self.train_model, self.unlabel_loader, self.pseudo_labels, self.device, self.dtype, batch_size = self.batch_size, criterion = nn.CrossEntropyLoss())
                
                # Select random subsets
                subsets = self.select_random_set()
                print("Greedy FL Start!")
                subset_count = 0
                for subset in subsets: 
                    if subset_count % 500 == 0:
                        print("Handling subset #", subset_count, " out of #", len(subsets))
                    gradient_data = self.gradients[subset].squeeze()
                    if gradient_data.size > 0:
                        gradient_data = gradient_data.reshape(gradient_data.shape[0], -1)
                    else:
                        continue
                    # Facility location function

                    fl_labels = self.pseudo_labels[subset] - torch.min(self.pseudo_labels[subset])
          
                    sub_coreset, sub_weights= Facility_Update.get_orders_and_weights(
                        20,
                        gradient_data,
                        "euclidean",
                        y=fl_labels.cpu().numpy(),
                        equal_num=False,
                        mode="sparse",
                        num_n=128,
                    )

                    sub_coreset_index = subset[sub_coreset] # Get the indices of the coreset
                    self.coreset.extend(sub_coreset_index.tolist()) # Add the coreset to the coreset list
                    self.weights.extend(sub_weights.tolist()) # Add the weights to the weights list
                    self.subsets.extend(subset)
                    subset_count += 1

                # update final coreset and weights
                if self.final_coreset is None:
                    # Directly assign the first dataset
                    self.final_coreset = Subset(self.unlabel_loader.dataset, self.coreset)
                else:
                    # Concatenate additional datasets
                    self.final_coreset = ConcatDataset([self.final_coreset, Subset(self.unlabel_loader.dataset, self.coreset)])
                # self.final_coreset = ConcatDataset([self.final_coreset, Subset(self.unlabel_loader.dataset, self.coreset)])
                for weight in self.weights:
                    self.final_weights.append(weight)
                    
                print("Greedy FL End!")

                # Update approx_loader
                self.approx_loader = DataLoader(Subset(self.unlabel_loader.dataset, self.coreset), batch_size=self.batch_size, shuffle=False, drop_last=True)
                # Get quadratic approximation
                self.get_quadratic_approximation()
                self.update_coreset = 0
                self.update_times += 1

            # Not to update coreset
            if self.update_coreset == 0:
                print("Not training coreset!")
                # Train iteratively with the coreset when approx error is within the threshold
                for i in range(50):
                    if(i % 10 == 0):
                        print("Training coreset #", i)
                    self.train_coreset()
                print("gf: ", self.gf)
                print("ggf: ", self.ggf)
                print("ggf_moment: ", self.ggf_moment)
                print("delta: ", self.delta)
                print("start_loss: ", self.start_loss)
                self.check_approx_error()
                if self.update_coreset == 1:
                    if(self.update_times == self.update_times_limit):
                        print("TRAINING COMPLETED!")
                        self.test_accuracy_with_weight()
                        self.test_accuracy_without_weight()
                    else:
                        # update unlabel_loader, remove coreset
                        all_indices = set(range(len(self.unlabel_loader.dataset)))
                        coreset_indices = set(self.coreset)
                        remaining_indices = list(all_indices - coreset_indices)
                        self.unlabel_loader = DataLoader(Subset(self.unlabel_loader.dataset, remaining_indices), batch_size=self.batch_size, shuffle=False, drop_last=True)
                        self.reset()


# --------------------------------
# Auxiliary functions        
            
    def check_approx_error(self):
        # calculate true loss    
        true_loss = 0 
        count = 0 
        subset_loader = DataLoader(Subset(self.unlabel_loader.dataset, self.subsets), batch_size=self.batch_size, shuffle=False, drop_last=True)
        optimizer = optim.Adam(self.train_model.parameters(), lr=self.lr)
        train_criterion = nn.CrossEntropyLoss()
        for approx_batch, (input, target) in enumerate(subset_loader):
            self.train_model.train()
            optimizer.zero_grad()
            input = input.to(self.device, dtype=self.dtype)
            pseudo_y = self.pseudo_labels[self.subsets[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]].to(self.device, dtype=self.dtype).squeeze().long()

            output,_ = self.train_model(input)

            loss = train_criterion(output, pseudo_y)
            
            loss.backward()
            optimizer.step()
            true_loss += loss.item() * input.size(0)
            count += input.size(0)
        true_loss = true_loss / count

        # self.train_model.train()
        approx_loss = torch.matmul(self.delta, self.gf) + self.start_loss
        approx_loss += 1/2 * torch.matmul(self.delta * self.ggf, self.delta)

        loss_diff = abs(true_loss - approx_loss.item())
        print("true loss: ", true_loss)
        print("approx loss: ", approx_loss)
        print("diff: ", loss_diff)

        #------------TODO--------------
        # ratio = loss_diff / true_loss
        # print("ratio: ", ratio)
        thresh = self.check_thresh_factor * true_loss                          
        #------------TODO--------------

        if loss_diff > thresh: 
            self.update_coreset = 1
            print("Need to reset coreset!")
            self.pseudo_labels= Train_model_trainer.psuedo_labeling(self.train_model, self.device, self.dtype, loader = self.unlabel_loader)
        else:
            self.update_coreset = 0
            print("No need to reset coreset!")

    def get_quadratic_approximation(self):
        self.start_loss = 0 
        train_criterion = nn.CrossEntropyLoss()
        self.weights = torch.tensor(self.weights, dtype=self.dtype)
        count = 0
        for approx_batch, (input, target) in enumerate(self.approx_loader):
            input = input.to(self.device, dtype=self.dtype)
            pseudo_y = self.pseudo_labels[self.coreset[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]].to(self.device, dtype=self.dtype).squeeze().long()
            weights = self.weights.clone().detach().to(device=self.device, dtype=self.dtype)
            # train coresets(with weights)
            output, _ = self.train_model(input)
            loss = train_criterion(output, pseudo_y)
            batch_weight = weights[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]
            loss = (loss * batch_weight).mean()
            self.train_model.zero_grad()
            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(momentum=True)
            if approx_batch == 0:
                self.gf = gf_tmp * self.batch_size
                self.ggf = ggf_tmp * self.batch_size
                self.ggf_moment = ggf_tmp_moment * self.batch_size
            else:
                self.gf += gf_tmp * self.batch_size
                self.ggf += ggf_tmp * self.batch_size
                self.ggf_moment += ggf_tmp_moment * self.batch_size

            self.start_loss += loss.item() * input.size(0) # each batch contributes to the total loss proportional to its size
            # self.start_loss = self.start_loss + loss.item()
            count += input.size(0)
        self.gf /= len(self.approx_loader.dataset)
        self.ggf /= len(self.approx_loader.dataset)
        self.ggf_moment /= len(self.approx_loader.dataset)
        self.start_loss = self.start_loss / count 
        self.delta = 0
        print("Quadratic approximation complete!")
            

    def train_coreset(self):
        train_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.train_model.parameters(), lr=self.lr)
        self.weights = torch.tensor(self.weights, dtype=self.dtype)
        for approx_batch, (input, target) in enumerate(self.approx_loader):
            optimizer.zero_grad()
            input = input.to(self.device, dtype=self.dtype)
            # pseudo_y = self.pseudo_labels[self.coreset[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]].to(self.device, dtype=self.dtype).squeeze().long()
            target = target.to(self.device, dtype=self.dtype).squeeze().long()

            self.train_model.train()

            weights = self.weights.clone().detach().to(device=self.device, dtype=self.dtype)
            output,_ = self.train_model(input)

            # loss = train_criterion(output, pseudo_y)
            loss = train_criterion(output, target)

            batch_weight = weights[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]
            loss = (loss * batch_weight).mean()
            
            self.train_model.zero_grad()
            loss.backward(create_graph=True)
            gf_current, _, _ = self.gradient_approx_optimizer.step(momentum=False)
            self.delta -= self.lr * gf_current  
            loss.backward()
            optimizer.step()
    


    def select_random_set(self):
        total_number = len(self.gradients)
        print("total_number: ", total_number)
        indices = np.arange(total_number) 
        np.random.shuffle(indices)
        self.subset_size = int(np.ceil(total_number / self.num_subsets))
        print("subset_size: ", self.subset_size)
        subsets = [indices[i * self.subset_size:(i + 1) * self.subset_size] for i in range(self.num_subsets)]
        return subsets



    def test_accuracy_without_weight(self):
        test_model = restnet_1d.build_model()
        print("Testing accuracy without weight!")
        unlabel_loader = self.unlabel_loader
        coreset = self.final_coreset
        weights = self.final_weights

        coreset_loader = DataLoader(coreset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_model.train()
        test_model = test_model.to(device=self.device)
        optimizer = optim.Adam(test_model.parameters(), lr=0.00001)
        criterion = torch.nn.CrossEntropyLoss()
        weights = torch.tensor(weights, dtype=self.dtype)
        weights = weights.clone().detach().to(device=self.device, dtype=self.dtype)
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            for t,(x,y) in enumerate(coreset_loader):
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype).squeeze().long()
                # batch_weight = weights[0 + t * 1200 : 1200 + t * 1200]
                optimizer.zero_grad()
                output, _ = test_model(x)
                loss = criterion(output,y)
                # loss = (loss * batch_weight).mean()
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
            for t, (inputs, targets) in enumerate(unlabel_loader):
                inputs = inputs.to(self.device, dtype=self.dtype)
                targets = targets.to(self.device, dtype=self.dtype).squeeze().long()
                # Forward pass to get outputs
                scores, _ = test_model(inputs)
                # Calculate the loss
                loss = criterion(scores, targets)
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
    


    def test_accuracy_with_weight(self):
        test_model = restnet_1d.build_model()
        print("Testing accuracy with weight!")
        unlabel_loader = self.unlabel_loader
        coreset = self.final_coreset
        weights = self.final_weights

        coreset_loader = DataLoader(coreset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_model.train()
        test_model = test_model.to(device=self.device)
        optimizer = optim.Adam(test_model.parameters(), lr=0.00001)
        criterion = torch.nn.CrossEntropyLoss()
        weights = torch.tensor(weights, dtype=self.dtype)
        weights = weights.clone().detach().to(device=self.device, dtype=self.dtype)
        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            for t,(x,y) in enumerate(coreset_loader):
                x = x.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype).squeeze().long()
                batch_weight = weights[0 + t * self.batch_size : self.batch_size + t * self.batch_size]
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
            for t, (inputs, targets) in enumerate(unlabel_loader):
                inputs = inputs.to(self.device, dtype=self.dtype)
                targets = targets.to(self.device, dtype=self.dtype).squeeze().long()
                # Forward pass to get outputs
                scores, _ = test_model(inputs)
                # Calculate the loss
                loss = criterion(scores, targets)
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
    

caller = main_trainer()
caller.main_loop()
# print("len coreset: ", len(coreset))
# print("len weights: ", len(weights))
# np.save('/vol/bitbucket/kp620/FYP/coreset.npy', coreset)
# np.save('/vol/bitbucket/kp620/FYP/weights.npy', weights)

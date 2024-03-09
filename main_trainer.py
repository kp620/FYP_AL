# --------------------------------
# Import area
import Test_cuda, Train_model_trainer, Train_model, Data_wrapper, Gradient_model_trainer, Gradient_model, Facility_Location, Approx_optimizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
# --------------------------------
# Main processing logic

class main_trainer():
    def __init__(self) -> None:
        self.device, self.dtype = Test_cuda.check_device() # Check if cuda is available
        # Model & Parameters
        self.train_model = Train_model.build_model() # Model used to train the data(M_0)
        self.batch_size = 128 # Batch size
        self.lr = 0.001 # Learning rate
        # Counter
        self.update_coreset = 1 # Decide if coreset needs to be updated
        self.update_times = 0 # Number of times the coreset has been updated
        self.update_times_limit = 10 # Maximum number of times the coreset can be updated (SELF_TUNE!!!)
        # Variables
        self.unlabel_loader = None # Set of unlabelled data
        self.approx_loader = None # Set of data used to approximate the loss
        self.pseudo_labels = [] # Store pseudo labels of unlabel_set
        self.gradients = [] # Store gradients of unlabel_set
        self.coreset = [] # Store coreset
        self.weights = [] # Store weights
        self.rs_rate = 0.05 # Percentage of the dataset to acuquire manual labelling
        self.num_subsets = 5000 # Number of random subsets
        self.subset_size = 0 # Size of each subset
        self.gf = 0 # Store gradient
        self.ggf = 0 # Store hutchinson trace
        self.ggf_moment = 0 # Store hutchinson trace moment
        self.delta = 0 # Store delta
        self.start_loss = 0 # Store start loss
        self.check_thresh_factor = 0.1 # Threshold factor (SELF_TUNE!!!)

    def reset(self):
        self.gradients = None
        self.approx_loader = None
        self.coreset = []
        self.weights = []
        self.gf = 0
        self.ggf = 0
        self.ggf_moment = 0
        self.delta = 0
        self.start_loss = 0

    # Given the initial dataset, select a subset of the dataset to train the initial model M_0
    def initial_training(self):
        # Load training data, acquire label and unlabel set using rs_rate
        # Divide the label set into training and validation set
        ini_train_loader, unlabel_set = Data_wrapper.process(batch_size=self.batch_size, rs_rate=self.rs_rate)
        # Update unlabel_loader
        self.unlabel_loader =  DataLoader(unlabel_set, batch_size=self.batch_size, shuffle=True)
        # Train the initial model over the label set
        Train_model_trainer.initial_train(ini_train_loader, self.train_model, self.device, self.dtype, criterion=nn.BCELoss(), learning_rate=self.lr)
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
                self.gradients = Train_model_trainer.gradient_train(self.train_model, self.unlabel_loader, self.pseudo_labels, self.device, self.dtype, batch_size = self.batch_size, criterion = nn.BCELoss())
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
                    sub_coreset, weights, _, _ = Facility_Location.facility_location_order(gradient_data, metric='euclidean', budget=10, weights=None, mode="dense", num_n=64)
                    sub_coreset = subset[sub_coreset]
                    self.coreset.extend(sub_coreset.tolist())
                    self.weights.extend(weights.tolist())
                    subset_count += 1
                print("Greedy FL End!")
                
                print("len(coreset): ", len(self.coreset))
                are_distinct = len(self.coreset) == len(set(self.coreset))
                print("whether distinctive? ", are_distinct)
                print("len(weights): ", len(self.weights))

                # Update approx_loader
                self.approx_loader = DataLoader(Subset(self.unlabel_loader.dataset, self.coreset), batch_size=self.batch_size, shuffle=False)
                # Get quadratic approximation
                self.get_quadratic_approximation()
                self.update_coreset = 0
                self.update_times += 1

            # Not to update coreset
            if self.update_coreset == 0:
                print("Not training coreset!")
                # Train iteratively with the coreset when approx error is within the threshold
                self.train_coreset()
                print("gf: ", self.gf)
                print("ggf: ", self.ggf)
                print("ggf_moment: ", self.ggf_moment)
                print("delta: ", self.delta)
                print("start_loss: ", self.start_loss)
                self.check_approx_error()
        return self.coreset, self.weights, self.train_model

# --------------------------------
# Auxiliary functions        
            
    def check_approx_error(self):
        # calculate true loss
        self.train_model.eval()
        true_loss = 0
        train_criterion = nn.BCELoss()

        with torch.no_grad():
            for approx_batch, (input, target) in enumerate(self.approx_loader):
                input = input.to(self.device, dtype=self.dtype)
                # target = target.to(self.device, dtype=self.dtype)
                pseudo_y = self.pseudo_labels[self.coreset[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]].to(self.device, dtype=self.dtype).unsqueeze(1)

                output = self.train_model(input)
                loss = train_criterion(output, pseudo_y)
                true_loss += loss.item()

        self.train_model.train()
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
            self.update_coreset = 1
            print("Need to reset coreset!")
            self.pseudo_labels= Train_model_trainer.psuedo_labeling(self.train_model, self.device, self.dtype, loader = self.unlabel_loader)
            self.reset()
        else:
            self.update_coreset = 0
            

    def train_coreset(self):
        gradient_approx_optimizer = Approx_optimizer.Adahessian(self.train_model.parameters())
        train_criterion = nn.BCELoss()
        optimizer = optim.Adam(self.train_model.parameters(), lr=self.lr)
        self.weights = torch.tensor(self.weights, dtype=self.dtype)
        for approx_batch, (input, target) in enumerate(self.approx_loader):
            optimizer.zero_grad()
            input = input.to(self.device, dtype=self.dtype)
            # target = target.to(self.device, dtype=self.dtype)
            pseudo_y = self.pseudo_labels[self.coreset[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]].to(self.device, dtype=self.dtype).unsqueeze(1)

            self.train_model.train()

            weights = self.weights.clone().detach().to(device=self.device, dtype=self.dtype)
            output = self.train_model(input)

            loss = train_criterion(output, pseudo_y)

            batch_weight = weights[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]
            loss = (loss * batch_weight).mean()
            
            self.train_model.zero_grad()
            loss.backward(create_graph=True)
            gf_current, _, _ = gradient_approx_optimizer.step(momentum=False)
            self.delta = self.delta - self.lr * gf_current
            
            
            loss.backward()
            optimizer.step()


    def get_quadratic_approximation(self):
        train_criterion = nn.BCELoss()
        gradient_approx_optimizer = Approx_optimizer.Adahessian(self.train_model.parameters())
        # self.coreset = self.coreset.astype(np.int64)
        self.weights = torch.tensor(self.weights, dtype=self.dtype)
        count = 0
        for approx_batch, (input, target) in enumerate(self.approx_loader):
            input = input.to(self.device, dtype=self.dtype)
            # target = target.to(self.device, dtype=self.dtype)
            pseudo_y = self.pseudo_labels[self.coreset[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]].to(self.device, dtype=self.dtype).unsqueeze(1)
            weights = self.weights.clone().detach().to(device=self.device, dtype=self.dtype)
            # train coresets(with weights)
            output = self.train_model(input)
            loss = train_criterion(output, pseudo_y)
            batch_weight = weights[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]
            loss = (loss * batch_weight).mean()
            self.train_model.zero_grad()
            # approximate with hessian diagonal
            loss.backward(create_graph=True)
            gf_tmp, ggf_tmp, ggf_tmp_moment = gradient_approx_optimizer.step(momentum=True)
            if approx_batch == 0:
                self.gf = gf_tmp * self.batch_size
                self.ggf = ggf_tmp * self.batch_size
                self.ggf_moment = ggf_tmp_moment * self.batch_size
            else:
                self.gf += gf_tmp * self.batch_size
                self.ggf += ggf_tmp * self.batch_size
                self.ggf_moment += ggf_tmp_moment * self.batch_size

            self.start_loss = self.start_loss + loss.item() * input.size(0) # each batch contributes to the total loss proportional to its size
            # self.start_loss = self.start_loss + loss.item()
            count += input.size(0)
        self.gf /= len(self.approx_loader.dataset)
        self.ggf /= len(self.approx_loader.dataset)
        self.ggf_moment /= len(self.approx_loader.dataset)
        self.start_loss = self.start_loss / count 
        print("Quadratic approximation complete!")


    def select_random_set(self):
        total_number = len(self.gradients)
        print("total_number: ", total_number)
        indices = np.arange(total_number) 
        np.random.shuffle(indices)
        self.subset_size = int(np.ceil(total_number / self.num_subsets))
        print("subset_size: ", self.subset_size)
        subsets = [indices[i * self.subset_size:(i + 1) * self.subset_size] for i in range(self.num_subsets)]
        return subsets






caller = main_trainer()
coreset , weights, model = caller.main_loop()
# print("shape(weights): ", weights.shape)
# torch.save(model.state_dict(), 'model_parameters.pth') # Save model parameters
# --------------------------------
# Import area
import test_cuda, Initial_Training, Initial_Model, Random_Sampling, gradient_trainer, gradient_model, Facility_Location, gradients
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
# --------------------------------
# Main processing logic

class main_trainer():
    def __init__(self) -> None:
        self.device, self.dtype = test_cuda.check_device() # Check if cuda is available
        # Model & Parameters
        self.train_model = Initial_Model.build_model() # Model used to train the data(M_0)
        self.gradient_model = gradient_model.build_model()# Model used to calculate gradients(M_G)
        self.batch_size = 128
        self.lr = 0.001
        # Counter
        self.update_coreset = 1 # Decide if coreset needs to be updated
        # Variables
        self.rs_rate = 0.05 # Percentage of the dataset to acuquire manual labelling
        self.unlabel_loader = None # Set of unlabelled data
        self.pseudo_labels = None # Store pseudo labels of unlabel_set
        self.gradients = None # Store gradients of unlabel_set
        self.num_subsets = 100 # Number of random subsets
        self.coreset = []
        self.weights = []
        self.gf = 0
        self.ggf = 0
        self.ggf_moment = 0
        self.delta = 0
        self.approx_loader = None
        self.start_loss = 0

        self.update_times = 0

    def reset(self):
        self.gradients = None
        self.coreset = []
        self.weights = []
        self.gf = 0
        self.ggf = 0
        self.ggf_moment = 0
        self.delta = 0
        self.approx_loader = None
        self.start_loss = 0

    # Given the initial dataset, select a subset of the dataset to train the initial model M_0
    def initial_training(self):
        ini_train_loader, ini_val_loader, unlabel_set = Random_Sampling.process(batch_size=self.batch_size, rs_rate=self.rs_rate)
        self.unlabel_loader =  DataLoader(unlabel_set, batch_size=self.batch_size, shuffle=True)
        Initial_Training.initial_train(ini_train_loader, ini_val_loader, self.train_model, self.device, self.dtype, criterion=nn.BCELoss(), learning_rate=self.lr)
        self.pseudo_labels = Initial_Training.psuedo_labeling(self.train_model, self.device, self.dtype, batch_size=self.batch_size)

    def main_loop(self):
        self.initial_training()
        while self.update_times < 10:
            # Update coreset!
            if self.update_coreset == 1:
                self.gradients = gradient_trainer.train_epoch(self.gradient_model, self.unlabel_loader, self.pseudo_labels, self.device, self.dtype, batch_size = self.batch_size, criterion = nn.CrossEntropyLoss())
                subsets = self.select_random_set()
                print("Facility Location Start!")
                subset_count = 0
                for subset in subsets: 
                    print("Handling subset #", subset_count)
                    gradient_data = self.gradients[subset].squeeze()
                    sub_coreset, weights, _, _ = Facility_Location.facility_location_order(gradient_data, metric='euclidean', budget=100, weights=None, mode="dense", num_n=64)
                    self.coreset.append(sub_coreset)
                    self.weights.append(weights)
                    subset_count += 1
                self.approx_loader = DataLoader(Subset(self.unlabel_loader.dataset, self.coreset), batch_size=self.batch_size, shuffle=False)
                self.get_quadratic_approximation()
                self.update_coreset = 0
                self.update_times += 1
            
            if self.update_coreset == 0:
                print("Not training coreset!")
                self.train_coreset()
                self.check_approx_error()

# --------------------------------
# Auxiliary functions        
            
    def check_approx_error(self):
        # calculate true loss
        self.train_model.eval()
        true_loss = 0

        with torch.no_grad():
            for approx_batch, (input, target) in enumerate(self.approx_loader):
                input = input.to(self.device, dtype=self.dtype)
                target = target.to(self.device, dtype=self.dtype)

                output = self.ini_model(input)
                loss = self.train_criterion(output, target)
                true_loss += loss.item()

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
            self.pseudo_labels= Initial_Training.psuedo_labeling(self.train_model, self.device, self.dtype, batch_size=self.batch_size)
            self.reset()
        else:
            self.update_coreset = 0
            

    def train_coreset(self):
        gradient_approx_optimizer = gradients.Adahessian(self.train_model.parameters())
        train_criterion = nn.BCELoss()
        optimizer = optim.Adam(self.train_model.parameters(), lr=self.lr)
        for approx_batch, (input, target) in enumerate(self.approx_loader):
            input = input.to(self.device, dtype=self.dtype)
            target = target.to(self.device, dtype=self.dtype)

            self.train_model.train()

            weights = self.weights.clone().detach().to(device=self.device, dtype=self.dtype)
            output = self.train_model(input)

            loss = train_criterion(output, target)

            batch_weight = weights[0 + approx_batch * self.batch_size : self.batch_size + approx_batch * self.batch_size]
            loss = (loss * batch_weight).mean()
            
            self.train_model.zero_grad()

            loss.backward(create_graph=True)
            gf_current, _, _ = gradient_approx_optimizer.step(momentum=False)
            self.delta = -self.lr * gf_current
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def get_quadratic_approximation(self):
        train_criterion = nn.BCELoss()
        gradient_approx_optimizer = gradients.Adahessian(self.train_model.parameters())
        self.coreset = self.coreset.astype(np.int64)
        self.weights = torch.tensor(self.weights, dtype=self.dtype)
        count = 0
        for approx_batch, (input, target) in enumerate(self.approx_loader):
            input = input.to(self.device, dtype=self.dtype)
            target = target.to(self.device, dtype=self.dtype)
            weights = self.weights.clone().detach().to(device=self.device, dtype=self.dtype)
            # train coresets(with weights)
            output = self.train_model(input)
            loss = train_criterion(output, target)
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

            self.start_loss = self.start_loss + loss.item() * input.size(0)
            count += input.size(0)
        self.gf /= len(self.approx_loader.dataset)
        self.ggf /= len(self.approx_loader.dataset)
        self.ggf_moment /= len(self.approx_loader.dataset)
        self.start_loss = self.start_loss / count #WHY?


    def select_random_set(self):
        total_number = len(self.gradients)
        indices = np.arange(total_number)
        np.random.shuffle(indices)
        subset_size = int(np.ceil(total_number / self.num_subsets))
        subsets = [indices[i * subset_size:(i + 1) * subset_size] for i in range(self.num_subsets)]
        return subsets

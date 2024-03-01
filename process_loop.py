import Initial_Training, Initial_Model
import test_cuda
import Random_Sampling
import gradient_trainer
import Facility_Location
import gradients

import torch
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Subset, DataLoader
import math

class main_Loop():
    def __init__(self):
        self.step = 0
        self.ini_model = Initial_Model.build_model()
        self.device, self.dtype = test_cuda.check_device()
        self.gradient = None
        self.subsets = None
        self.reset_coreset = 0
        self.delta = 0
        self.current_loss = 0
        self.learning_rate = 0.001

        self.gf = None
        self.ggf = None
        self.ggf_moment = None

        self.gradient_approx_optimizer = gradients.Adahessian(self.model.parameters())


    def loop(self):

        # Get initial model and pseudo labels
        ini_train_loader, ini_val_loader, unlabel_set =Random_Sampling.process(batch_size=128)
        if os.path.exists('initial_model.pth'):
            print("Initial model already exists!")
        else: 
            Initial_Training.initial_train(ini_train_loader, ini_val_loader, self.ini_model, self.device, self.dtype)
            print("Initial model created!")
        pseudo_labels, unlabel_loader = Initial_Training.psuedo_labeling(unlabel_set, self.device, self.dtype, batch_size=128)

        print("pseudo_labels shape: ", pseudo_labels.shape)
        print("pseudo_labels: ", pseudo_labels)
        print("unlabel_loader: ", unlabel_loader)

        
        # # Into the main loop!
        # if self.step == 0:
        #     # Train the model with the pseudo labels & Calculate the gradients
        #     gradients = self.gradient_calculator(unlabel_loader, pseudo_labels)
        #     self.gradients = np.concatenate(gradients, axis=0)
        #     print("Full gradients shape: ", self.gradients.shape)
        #     self.step += 1

        
        # # Define random subsets 
        # num_subsets = 1000
        # self.subsets = self.select_random_set(num_subsets, self.gradients)

        # # Facility Location
        # for subset in self.subsets: 
        #     gradient_data = self.gradients[subset].squeeze()
        #     sub_coreset, weights, _, _ = Facility_Location.facility_location_order(gradient_data, metric='euclidean', budget=100, weights=None, mode="dense", num_n=64)
        #     self.coreset = np.append(sub_coreset)
        #     self.weights = np.append(weights)


        # while self.reset_coreset == 0:
        #     # get quadratic approximation 
        #     gf, ggf, ggf_moment = self._get_quadratic_approximation(self.coreset, self.weights)

        #     self.delta = -self.learning_rate * gf
        #     # check approximation error
        #     self._check_approx_error(gf, ggf, ggf_moment)

        # # if self.reset_coreset == 1:
        #     # Need to reset coreset
            





    def _get_quadratic_approximation(self, coreset, weights, batch_size=128):
        
        approx_loader = DataLoader(Subset(self.unlabel_loader.dataset, coreset), batch_size=batch_size, shuffle=False)

        self.start_loss = 0

        for approx_batch, (input, target) in enumerate(approx_loader):
            input = input.to(self.device, dtype=self.dtype)
            target = target.to(self.device, dtype=self.dtype)

            # train with weights

            # train coresets(with weights)
            output = self.model(input)
                
            loss = self.train_criterion(output, target)
            batch_weight = weights[0+approx_batch*batch_size : batch_size+approx_batch*batch_size]
            loss = (loss * batch_weight).mean()

            self.model.zero_grad()

            # approximate with hessian diagonal
            loss.backward(create_graph=True)


            gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(self.model.parameters(), momentum=True)

            if approx_batch == 0:
                self.gf = gf_tmp * batch_size
                self.ggf = ggf_tmp * batch_size
                self.ggf_moment = ggf_tmp_moment * batch_size
            else:
                self.gf += gf_tmp * batch_size
                self.ggf += ggf_tmp * batch_size
                self.ggf_moment += ggf_tmp_moment * batch_size

            self.start_loss = loss.item()
 
        self.gf /= len(approx_loader.dataset)
        self.ggf /= len(approx_loader.dataset)
        self.ggf_moment /= len(approx_loader.dataset)


    def _check_approx_error(self):
        # calculate true loss
        
        model = self.model

        self.model.eval()

        self.train_output = np.zeros((len(self.train_dataset), self.args.num_classes))

        with torch.no_grad():
            for _, (data, _, data_idx) in enumerate(self.train_val_loader):
                data = data.to(self.args.device)

                output = self.model(data)

                self.train_output[data_idx] = output.cpu().numpy()

        self.model.train()




        true_loss = self.val_criterion(
            torch.from_numppy(self.train_output[self.random_sets]),
            torch.from_numppy(self.psuedo_label[self.random_sets]),
        )

        approx_loss = torch.matmul(self.delta, self.gf) + self.start_loss
        approx_loss += 1 / 2 * torch.matmul(self.delta * self.ggf, self.delta)

        loss_diff = abs(true_loss - approx_loss.item())
        thresh = self.args.check_thresh_factor * self.true_loss

        if loss_diff > thresh:
            self.reset_coreset = 1
            print("Reset Coreset!")



    def gradient_calculator(self, loader, pseudo_labels):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        gradients = gradient_trainer.train_epoch(self.ini_model, optimizer, criterion, loader, self.device, self.dtype, pseudo_labels)
        return gradients 
    

    def select_random_set(num_subsets, gradients):
        total_number = len(gradients)
        indices = np.arange(total_number)
        np.random.shuffle(indices)
        subset_size = int(np.ceil(total_number / num_subsets))
        subsets = [indices[i * subset_size:(i + 1) * subset_size] for i in range(num_subsets)]
        return subsets
    
main_Loop.loop()
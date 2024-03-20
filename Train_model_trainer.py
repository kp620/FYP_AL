import Train_model, restnet_1d
import torch
import torch.nn as nn
import torch.nn.functional as F


def initial_train(train_loader, model, device, dtype, criterion, learning_rate):
    restnet_1d.train_epoch(model, train_loader, device, dtype, criterion=criterion, learning_rate=learning_rate)
    print("Initial training complete!")

def gradient_train(model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion):
    gradients = restnet_1d.gradient_train_epoch(model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion)
    print("Gradient training complete!")
    return gradients

def psuedo_labeling(model, devc, dtype, loader):
    pseudo_labels = restnet_1d.psuedo_labeling(model, devc, dtype, loader)
    print("Pseudo labeling complete!")
    return pseudo_labels

    



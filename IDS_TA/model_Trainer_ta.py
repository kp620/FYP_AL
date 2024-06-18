"""
Model trainer module for training the model with the given data.
"""

import restnet_1d_dynamic as restnet_1d

def initial_train(train_loader, model, device, dtype, criterion, learning_rate, classifer, optimizer):
    print("Initial training started...")
    restnet_1d.train(model, train_loader, device, dtype, criterion=criterion, learning_rate=learning_rate, classifer=classifer, optimizer=optimizer)
    print("Initial training complete!")

def gradient_train(traning_step, model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion, classifer):
    print("Gradient trainining started at step: ", traning_step)
    gradients = restnet_1d.gradient_train(model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion, classifer=classifer)
    print("Gradient training complete!")
    return gradients

def psuedo_labeling(model, devc, dtype, loader, classifer):
    print("Pseudo labeling started...")
    pseudo_labels = restnet_1d.psuedo_labeling(model, devc, dtype, loader, classifer=classifer)
    print("Pseudo labeling complete!")
    return pseudo_labels

    



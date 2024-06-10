"""
Model trainer module for training the model with the given data.
"""

import restnet_1d

# Initial training
def initial_train(train_loader, model, device, dtype, criterion, learning_rate):
    print("Initial training started...")
    restnet_1d.train(model, train_loader, device, dtype, criterion=criterion, learning_rate=learning_rate)
    print("Initial training complete!")

# Measure the gradients
def gradient_train(traning_step, model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion):
    print("Gradient trainining started at step: ", traning_step)
    gradients = restnet_1d.gradient_train(model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion)
    print("Gradient training complete!")
    return gradients

# Pseudo labeling
def psuedo_labeling(model, devc, dtype, loader):
    print("Pseudo labeling started...")
    pseudo_labels = restnet_1d.psuedo_labeling(model, devc, dtype, loader)
    print("Pseudo labeling complete!")
    return pseudo_labels

    



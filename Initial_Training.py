import Initial_Model
import torch
import torch.nn as nn


def initial_train(train_loader, val_loader, model, device, dtype, criterion, learning_rate):
    Initial_Model.train_epoch(model, train_loader, val_loader, device, dtype, criterion=criterion, learning_rate=learning_rate)
    print("Initial training complete!")

def psuedo_labeling(model, devc, dtype, batch_size = 128):
    pseudo_labels = []
    # correct_pseudo_y = 0
    # total_correct_pseudo_y = 0
    with torch.no_grad():  # We don't need to compute gradients
        for x, real_y in unlabel_loader:
            x = x.to(devc, dtype=dtype)  # Move data to the appropriate device
            scores = model(x).squeeze()  # Get model predictions
        
            # Get the predicted classes (for classification tasks)
            pred_y = scores > 0.5
            # correct_pseudo_y += torch.sum(pred_y == real_y.data.to(devc, dtype=dtype).squeeze()).item()
            # total_correct_pseudo_y += real_y.size(0)

            pseudo_labels.append(pred_y.cpu())  # Store the predictions

    # pseudo_accuracy = correct_pseudo_y / total_correct_pseudo_y
    # print("correct pseudo labels: ", correct_pseudo_y)
    # print("total pseudo labels: ", total_correct_pseudo_y)
    # print("pseudo labelling accuracy: ", pseudo_accuracy)

    # Concatenate the list of labels into a single tensor
    pseudo_labels = torch.cat(pseudo_labels)
    print("Pseudo labeling complete!")
    return pseudo_labels

    



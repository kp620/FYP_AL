import Initial_Model
import torch
import torch.nn as nn
import os


def initial_train(train_loader, val_loader, model, devc, dtype):
    Initial_Model.train_epoch(model, train_loader, val_loader, devc, dtype, nn.BCELoss())
    torch.save(model, 'initial_model.pth')
    print("Model saved!")

def psuedo_labeling(unlabel_set, devc, dtype, batch_size = 128):
    model = torch.load('initial_model.pth')
    unlabel_loader = torch.utils.data.DataLoader(unlabel_set, batch_size=batch_size, shuffle=True)

    pseudo_labels = []
    correct_pseudo_y = 0
    total_correct_pseudo_y = 0
    with torch.no_grad():  # We don't need to compute gradients
        for x, real_y in unlabel_loader:
            x = x.to(devc, dtype=dtype)  # Move data to the appropriate device
            scores = model(x).squeeze()  # Get model predictions
        
            # Get the predicted classes (for classification tasks)
            pred_y = scores > 0.5

            correct_pseudo_y += torch.sum(pred_y == real_y.data.to(devc, dtype=dtype).squeeze()).item()
            total_correct_pseudo_y += real_y.size(0)

            pseudo_labels.append(pred_y.cpu())  # Store the predictions

    pseudo_accuracy = correct_pseudo_y / total_correct_pseudo_y
    print("correct pseudo labels: ", correct_pseudo_y)
    print("total pseudo labels: ", total_correct_pseudo_y)
    print("pseudo labelling accuracy: ", pseudo_accuracy)

    # Concatenate the list of labels into a single tensor
    pseudo_labels = torch.cat(pseudo_labels)
    return pseudo_labels, unlabel_loader

    



import numpy as np
import torch
import torch.nn.functional as F
import subprocess

def cal_uncertainties(model, unlabel_loader, device, dtype):
    uncertainties = []
    model.eval()
    model = model.to(device=device)
    for input, target, idx in unlabel_loader:
        input = input.to(device=device, dtype=dtype)
        with torch.no_grad():
            outputs, _ = model(input)
            probabilities = F.softmax(outputs, dim=1)
            log_probs = torch.log(probabilities + 1e-9)  # epsilon for numerical stability
            uncertainty = -torch.sum(probabilities * log_probs, dim=1)
            uncertainties.append(uncertainty.cpu().numpy())
    uncertainties = np.concatenate(uncertainties, axis=0)
    model.train()
    return uncertainties


def cal_similarities(eigenv, label_loader, unlabel_loader, sigma, device, dtype):
    # Collect all label features into a single tensor
    label_features = []
    for data, _ ,_ in label_loader:
        features = data.squeeze(1).to(device=device, dtype=dtype)  # Adjust processing as needed
        features = torch.mm(features, eigenv) # NEED TO TEST SIZE!!!!
        label_features.append(features)
    label_features = torch.cat(label_features, dim=0)

    similarities = []
    for inputs, _, _ in unlabel_loader:
        inputs = inputs.squeeze(1).to(device=device, dtype=dtype)
        inputs = torch.mm(inputs, eigenv) # NEED TO TEST SIZE!!!!
        # Expand dimensions for broadcasting and calculate distances
        distances = torch.cdist(inputs, label_features)
        min_distances = torch.min(distances, dim=1).values
        similarity = torch.exp(-(min_distances ** 2) / (2 * sigma ** 2))
        similarities.append(similarity.cpu().numpy())
    
    return np.concatenate(similarities)

def continuous_states(eigenv, label_loader, unlabel_loader, model, device, dtype, alpha, sigma):
    uncertainties = cal_uncertainties(model, unlabel_loader, device, dtype)
    similarities = cal_similarities(eigenv, label_loader, unlabel_loader, sigma, device, dtype)
    continuous_states = alpha * uncertainties + (1-alpha) * (1 - similarities)
    return continuous_states
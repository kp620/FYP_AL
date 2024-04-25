import numpy as np
import torch
import torch.nn.functional as F

def uncertainties(model, unlabel_loader, device, dtype):
    uncertainties = []
    model.eval()
    model = model.to(device=device)
    for input, target, idx in unlabel_loader:
        input = input.to(device=device, dtype=dtype)
        output, _ = model(input)
        probabilities = F.softmax(output, dim=1)
        uncertainty = -torch.sum(probabilities * torch.log(probabilities), dim=1)
        uncertainties.append(uncertainty.cpu().numpy())
    uncertainties = np.concatenate(uncertainties, axis=0)
    model.train()
    return uncertainties

def similarities(label_loader, unlabel_loader, sigma, device, dtype):
    similarities = []
    for input, target, idx in unlabel_loader:
        input = input.to(device=device, dtype=dtype)
        nearest_label_sample = min(label_loader.dataset.dataset, key=lambda l: np.linalg.norm(input - l))
        distance = np.linalg.norm(input - nearest_label_sample)
        similarity = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        similarities.append(similarity)
    similarities = np.concatenate(similarities, axis=0)
    return similarities

def continuous_states(final_coreset, unlabel_loader, model, device, dtype, alpha, beta, sigma):
    uncertainties = uncertainties(model, unlabel_loader, device, dtype)
    similarities = similarities(final_coreset, unlabel_loader, sigma, device, dtype)
    continuous_states = {}
    for i in range(len(unlabel_loader.dataset)):
        continuous_states[i] = alpha * uncertainties[i] + beta * (1 - similarities[i])
    return continuous_states
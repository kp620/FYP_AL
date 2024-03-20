import torch
import torch.nn.functional as F

# Example model output (logits) for a single sample
logits = torch.tensor([[2.0, -1.0]])

# Apply softmax to convert logits to probabilities
probabilities = F.softmax(logits, dim=1)
print("Probabilities:", probabilities)

# Determine the class with the highest probability
_, pseudo_label = torch.max(probabilities, dim=1)
print("Pseudo-label:", pseudo_label.item())  # Use .item() to get the value as a Python integer

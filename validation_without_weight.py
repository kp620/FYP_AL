import restnet_1d, Test_cuda
import math
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pickle
import pandas as pd
from torch.utils.data import TensorDataset


def load_data():
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    x_data = pd.read_csv(f'{data_dic_path}/x_data_iid.csv').astype(float)
    y_data = pd.read_csv(f'{data_dic_path}/y_data_iid.csv').astype(float)
    print('Training Data Loaded!')
    print('Total Length: ', len(x_data))
    x_data = torch.from_numpy(x_data.values).unsqueeze(1)
    y_data = torch.from_numpy(y_data.values)
    full_dataset = TensorDataset(x_data, y_data)
    return full_dataset


coreset = np.load('/vol/bitbucket/kp620/FYP/coreset.npy')
weights = np.load('/vol/bitbucket/kp620/FYP/weights.npy')
with open('/vol/bitbucket/kp620/FYP/not_chosen_indices.pkl', 'rb') as f:
    not_chosen_indices = pickle.load(f)
print("coreset length: ", len(coreset))
print("weights length: ", len(weights))
print("not_chosen_indices length: ", len(not_chosen_indices))


full_dataset = load_data()
unlabel_set = Subset(full_dataset, list(not_chosen_indices))

model = restnet_1d.build_model()
device, dtype = Test_cuda.check_device()

unlabel_loader = torch.utils.data.DataLoader(unlabel_set, batch_size=64, shuffle=False)
coreset_loader = DataLoader(Subset(unlabel_loader.dataset, coreset), batch_size=1200, shuffle=False, drop_last=True)

all_indices = set(range(len(unlabel_loader.dataset)))
coreset_indices = set(coreset)
remaining_indices = list(all_indices - coreset_indices)
remaining_subset = Subset(unlabel_loader.dataset, remaining_indices)
remaining_loader = DataLoader(remaining_subset, batch_size=1200, shuffle=False, drop_last=True)

# Train the model with coreset 
model.train()
model = model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()
weights = torch.tensor(weights, dtype=dtype)
weights = weights.clone().detach().to(device=device, dtype=dtype)
    # Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for t,(x,y) in enumerate(coreset_loader):
        model.train()
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype).squeeze().long()
        # batch_weight = weights[0 + t * 1200 : 1200 + t * 1200]
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output,y)
        # loss = (loss * batch_weight).mean()
        loss.backward()
        optimizer.step()  
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Test the model
model.eval()
model = model.to(device=device)
correct_predictions = 0
total_predictions = 0
# Disable gradient calculations
with torch.no_grad():
    for t, (inputs, targets) in enumerate(remaining_loader):
        inputs = inputs.to(device, dtype=dtype)
        targets = targets.to(device, dtype=dtype).squeeze().long()
        # Forward pass to get outputs
        scores, _ = model(inputs)
        # Calculate the loss
        loss = criterion(scores, targets)

        probabilities = F.softmax(output, dim=1)
        _, pseudo_label = torch.max(probabilities, dim=1)

        # Count correct predictions
        correct_predictions += torch.sum(pseudo_label == targets.data).item()
        total_predictions += targets.size(0)
# Calculate average loss and accuracy
test_accuracy = correct_predictions / total_predictions
print("correct predictions: ", correct_predictions)
print("total_predictions: ", total_predictions)
# Return results
print(f'Test Accuracy: {test_accuracy:.2f}')


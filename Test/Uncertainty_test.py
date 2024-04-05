import torch
import Test_cuda, restnet_1d_binary
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

device, dtype = Test_cuda.check_device()

def load_data():
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    x_data = pd.read_csv(f'{data_dic_path}/x_data_iid.csv').astype(float)
    y_data = pd.read_csv(f'{data_dic_path}/y_data_iid.csv').astype(float)
    print('Training Data Loaded!')
    print('Total Length: ', len(x_data))
    x_data = torch.from_numpy(x_data.values).unsqueeze(1)
    y_data = torch.from_numpy(y_data.values)
    return x_data, y_data

def uncertainty_sampling(model, x_data, y_data, selection_rate, device = device, dtype = dtype, batch_size = 128):
    # Make sure the model is in evaluation mode
    model.eval()
    model = model.to(device=device)

    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []

    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(loader):
            x_batch = x_batch.to(device=device, dtype=dtype)
            outputs, _ = model(x_batch)
            probs, _ = torch.max(torch.softmax(outputs, dim=1), dim=1)
            all_probs.append(probs.cpu())   

    # Concatenate all batch probabilities and sort them to find the most uncertain samples
    all_probs = torch.cat(all_probs)
    num_samples = len(all_probs)
    num_select = int(num_samples * selection_rate)
    
    uncertainty_indices = all_probs.argsort()[:num_select]
    
    # Use these indices to select the samples
    x_selected = x_data[uncertainty_indices]
    y_selected = y_data[uncertainty_indices]

    not_selected_mask = ~torch.isin(torch.arange(num_samples), uncertainty_indices)
    x_not_selected = x_data[not_selected_mask]
    y_not_selected = y_data[not_selected_mask]

    return x_selected, y_selected, x_not_selected, y_not_selected


model = restnet_1d_binary.build_model()
model.to(device=device)
x_data, y_data = load_data()

x_selected, y_selected, x_not_selected, y_not_selected = uncertainty_sampling(model, x_data, y_data, 0.015)
print("x_selected length: ", len(x_selected))

train_loader = DataLoader(TensorDataset(x_selected, y_selected), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(x_not_selected, y_not_selected), batch_size=128, shuffle=True)


# # Train the model with coreset 
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()
    # Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for t,(x,y) in enumerate(train_loader):
        model.train()
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype).squeeze().long()
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()  
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Test the model
model.eval()
correct_predictions = 0
total_predictions = 0
# Disable gradient calculations
with torch.no_grad():
    for t, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device, dtype=dtype)
        targets = targets.to(device, dtype=dtype).squeeze().long()
        # Forward pass to get outputs
        scores, _ = model(inputs)
        # Calculate the loss
        loss = criterion(scores, targets)

        probabilities = F.softmax(scores, dim=1)
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
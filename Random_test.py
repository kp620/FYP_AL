import torch
import Test_cuda, restnet_1d
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

def random_sampling(x_data, y_data, selection_rate):
    num_samples = x_data.shape[0]
    num_select = int(num_samples * selection_rate)
    all_indices = torch.randperm(num_samples)
    selected_indices = all_indices[:num_select]
    x_selected = x_data[selected_indices]
    y_selected = y_data[selected_indices]
    not_selected_indices = all_indices[num_select:]
    x_not_selected = x_data[not_selected_indices]
    y_not_selected = y_data[not_selected_indices]

    return x_selected, x_not_selected, y_selected, y_not_selected


model = restnet_1d.build_model()
x_data, y_data = load_data()

x_selected, x_not_selected, y_selected, y_not_selected = random_sampling(x_data, y_data, 0.15)

train_loader = DataLoader(TensorDataset(x_selected, y_selected), batch_size=1200, shuffle=True, drop_last=True)
test_loader = DataLoader(TensorDataset(x_not_selected, y_not_selected), batch_size=1200, shuffle=True, drop_last=True)


# Train the model with coreset 
model.train()
model = model.to(device=device)
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
model = model.to(device=device)
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
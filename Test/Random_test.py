import torch
import sys 
sys.path.append('../')
import test_Cuda, restnet_1d, restnet_1d_multi
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device, dtype = test_Cuda.check_device()

def load_data():
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    x_data = pd.read_csv(f'{data_dic_path}/x_data_iid_multiclass.csv').astype(float)
    y_data = pd.read_csv(f'{data_dic_path}/y_data_iid_multiclass.csv').astype(float)
    print('Test Data Loaded!')
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


model = restnet_1d_multi.build_model()
model.to(device=device)
x_data, y_data = load_data()

x_selected, x_not_selected, y_selected, y_not_selected = random_sampling(x_data, y_data, 0.11)
print("x_selected length: ", len(x_selected))

train_loader = DataLoader(TensorDataset(x_selected, y_selected), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(x_not_selected, y_not_selected), batch_size=128, shuffle=True)


# # Train the model with coreset 
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()
    # Training loop
num_epochs = 50
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

model.eval()
predictions = []
targets = []
# Disable gradient calculations
with torch.no_grad():
    for batch, (input, target) in enumerate(test_loader):
        input = input.to(device, dtype=dtype)
        target = target.to(device, dtype=dtype).squeeze().long()
        output, _ = model(input)
        probabilities = F.softmax(output, dim=1)
        _, pseudo_label = torch.max(probabilities, dim=1)
        predictions.extend(pseudo_label.cpu().numpy())
        targets.extend(target.cpu().numpy())
predictions = np.array(predictions)
targets = np.array(targets)

accuracy = accuracy_score(targets, predictions)
# precision = precision_score(targets, predictions, average='weighted')
# recall = recall_score(targets, predictions, average='weighted')
# f1 = f1_score(targets, predictions, average='weighted')
# print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

# precision = precision_score(targets, predictions, average='micro')
# recall = recall_score(targets, predictions, average='micro')
# f1 = f1_score(targets, predictions, average='micro')
# print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

precision = precision_score(targets, predictions, average='macro')
recall = recall_score(targets, predictions, average='macro')
f1 = f1_score(targets, predictions, average='macro')
print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
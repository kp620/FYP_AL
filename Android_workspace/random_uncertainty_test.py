import torch
import sys 
sys.path.append('../')
import test_Cuda, restnet_1d_android
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import copy
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument("--operation_type",  choices=['iid', 'time-aware'], help='Specify operation type: "iid" or "time-aware"')
parser.add_argument("--class_type", choices=['binary', 'multi'], help='Specify class type: "binary" or "multi"')
parser.add_argument("--budget", type=float, help='Specify the budget')
args = parser.parse_args()

device, dtype = test_Cuda.check_device()

def load_file(file):
        data = np.load(file)
        x_train = data['X_train']
        y_train = data['y_train']

        y_train_binary = np.where(y_train == 0, 0, 1) # Convert to binary

        y_mal_family = data['y_mal_family']
        return x_train, y_train_binary, y_mal_family
    

def load_data():
    print("Loading data...")
    directory = '/vol/bitbucket/kp620/FYP/Android_workspace/data/gen_apigraph_drebin'
    x_train = []
    y_train = []
    y_mal_family = []
    for file in os.listdir(directory):
        if file.endswith('.npz'):
            command = "echo 'Loading file: " + file + "'"
            subprocess.call(command, shell=True)
            x, y, y_mal = load_file(os.path.join(directory, file))
            x_train.append(x)
            y_train.append(y)
            y_mal_family.append(y_mal)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_data = pd.DataFrame(x_train).astype(float)
    y_data = pd.DataFrame(y_train).astype(float)
    print('Training Data Loaded!')
    print('Total Length: ', len(x_data))
    x_data = torch.from_numpy(x_data.values).unsqueeze(1)
    y_data = torch.from_numpy(y_data.values).squeeze()
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

def uncertainty_sampling(model, x_data, y_data, selection_rate, device = device, dtype = dtype, batch_size = 1024):
    # Make sure the model is in evaluation mode
    model.eval()
    # Ensure x_data and y_data are on the same device as the model
    x_data = x_data.to(device, dtype = dtype)
    y_data = y_data.to(device, dtype = dtype).long()
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
    return x_selected, x_not_selected, y_selected, y_not_selected

def train_model(model, train_loader, device, dtype, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for t,(x,y) in enumerate(train_loader):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype).long()
            optimizer.zero_grad()
            output, _ = model(x)
            loss = criterion(output,y)
            loss.backward()
            optimizer.step()  
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return model

def eval_model(model, test_loader, device, dtype):
    model.eval()
    predictions = []
    targets = []
    # Disable gradient calculations
    with torch.no_grad():
        for batch, (input, target) in enumerate(test_loader):
            input = input.to(device, dtype=dtype)
            target = target.to(device, dtype=dtype).long()
            output, _ = model(input)
            probabilities = F.softmax(output, dim=1)
            _, pseudo_label = torch.max(probabilities, dim=1)
            predictions.extend(pseudo_label.cpu().numpy())
            targets.extend(target.cpu().numpy())
    predictions = np.array(predictions)
    targets = np.array(targets)

    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')
    f1 = f1_score(targets, predictions, average='macro')
    confusion_matrix_result = confusion_matrix(targets, predictions)
    print(f'Macro Precision: {precision}, Macro Recall: {recall}, Macro F1: {f1}')
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    print(f'Weighted Precision: {precision}, Weighted Recall: {recall}, Weighted F1: {f1}')
    print(f'Confusion Matrix: {confusion_matrix_result}')



# ----------------- Main -----------------
x_data, y_data = load_data()

AL_iter = 10
selection_budget = args.budget
num_epochs = 50

x_train, x_data, y_train, y_data = random_sampling(x_data, y_data, (selection_budget/AL_iter))

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=1024, shuffle=True)
test_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=1024, shuffle=True)
model = restnet_1d_android.build_model("multi").to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()

command = 'echo "initial training"'
subprocess.call(command, shell=True)
model = train_model(model, train_loader, device, dtype, criterion, optimizer, num_epochs)
command = 'echo "initial evaluation"'
subprocess.call(command, shell=True)
eval_model(model, test_loader, device, dtype)

us_model = copy.deepcopy(model)
rs_model = copy.deepcopy(model)

def US_Model(us_model, x_data, y_data, x_train, y_train):
    command = 'echo "uncertainty sampling"'
    subprocess.call(command, shell=True)
    for i in range(0, AL_iter):
        command = f'echo "Update {i+1}"'
        subprocess.call(command, shell=True)
        x_selected, x_data, y_selected, y_data = uncertainty_sampling(us_model, x_data, y_data, selection_budget/AL_iter)
        x_train = torch.cat((x_train, x_selected.to('cpu')), dim=0)
        y_train = torch.cat((y_train, y_selected.to('cpu')), dim=0)  
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=1024, shuffle=True)
        us_model = train_model(us_model, train_loader, device, dtype, criterion, optimizer, num_epochs)
        command = 'echo "length of x_train: "' + str(len(x_train))
        subprocess.call(command, shell=True)
    print("length of x_train: ", len(x_train))
    test_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=1024, shuffle=True)
    command = 'echo "Evaluation"'
    subprocess.call(command, shell=True)
    eval_model(us_model, test_loader, device, dtype)

def RS_Model(rs_model, x_data, y_data, x_train, y_train):
    for i in range(0, AL_iter):
        command = 'echo "random sampling"'
        subprocess.call(command, shell=True)
        command = f'echo "Update {i+1}"'
        subprocess.call(command, shell=True)
        x_selected, x_data, y_selected, y_data = random_sampling(x_data, y_data, selection_budget/AL_iter)
        x_train = torch.cat((x_train, x_selected.to('cpu')), dim=0)
        y_train = torch.cat((y_train, y_selected.to('cpu')), dim=0)      
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=1024, shuffle=True)     
        rs_model = train_model(rs_model, train_loader, device, dtype, criterion, optimizer, num_epochs)
        command = 'echo "length of x_train: "' + str(len(x_train))
        subprocess.call(command, shell=True)          
    print("length of x_train: ", len(x_train))
    test_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=1024, shuffle=True)
    command = 'echo "Evaluation"'
    subprocess.call(command, shell=True)
    eval_model(rs_model, test_loader, device, dtype)


US_Model(us_model, x_data, y_data, x_train, y_train)
print("#-----------------------------------------------------#")
RS_Model(rs_model, x_data, y_data, x_train, y_train)

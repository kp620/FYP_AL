import torch
import Test_cuda, restnet_1d_binary
import pandas as pd
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import copy

device, dtype = Test_cuda.check_device()

def load_data():
    print("Loading data...")
    data_dic_path = "/vol/bitbucket/kp620/FYP/dataset"
    x_data = pd.read_csv(f'{data_dic_path}/x_data_iid_binary.csv').astype(float)
    y_data = pd.read_csv(f'{data_dic_path}/y_data_iid_binary.csv').astype(float)
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

    return x_selected, x_not_selected, y_selected, y_not_selected

def train_model(model, train_loader, device, dtype, criterion, optimizer, num_epochs):
    model.train()
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
    return model

def eval_model(model, test_loader, device, dtype, criterion):
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
    return test_accuracy


x_data, y_data = load_data()

AL_iter = 10
selection_budget = 0.013
num_epochs = 50


x_train, x_data, y_train, y_data = random_sampling(x_data, y_data, (selection_budget/AL_iter))
print("length of x_train: ", len(x_train))
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=128, shuffle=True)
model = restnet_1d_binary.build_model().to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()
model = train_model(model, train_loader, device, dtype, criterion, optimizer, num_epochs)
eval_model(model, test_loader, device, dtype, criterion)

us_model = copy.deepcopy(model)
rs_model = copy.deepcopy(model)


def RS_Model(rs_model, x_data, y_data, x_train, y_train):
    print("Random Sampling")
    for _ in range(0, AL_iter):
        print(f' Update {_+1}')
        x_selected, x_data, y_selected, y_data = random_sampling(x_data, y_data, selection_budget/AL_iter)
        x_train = torch.cat((x_train, x_selected.to('cpu')), dim=0)
        y_train = torch.cat((y_train, y_selected.to('cpu')), dim=0)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
        rs_model = train_model(rs_model, train_loader, device, dtype, criterion, optimizer, num_epochs)

    print("length of x_train: ", len(x_train))

    test_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=128, shuffle=True)
    eval_model(rs_model, test_loader, device, dtype, criterion)



def US_Model(us_model, x_data, y_data, x_train, y_train):
    print("Uncertainty Sampling")
    for _ in range(0, AL_iter):
        print(f' Update {_+1}')
        x_selected, x_data, y_selected, y_data = uncertainty_sampling(us_model, x_data, y_data, selection_budget/AL_iter)
        x_train = torch.cat((x_train, x_selected.to('cpu')), dim=0)
        y_train = torch.cat((y_train, y_selected.to('cpu')), dim=0)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
        us_model = train_model(us_model, train_loader, device, dtype, criterion, optimizer, num_epochs)

    print("length of x_train: ", len(x_train))

    test_loader = DataLoader(TensorDataset(x_data, y_data), batch_size=128, shuffle=True)
    eval_model(rs_model, test_loader, device, dtype, criterion)


US_Model(us_model, x_data, y_data, x_train, y_train)
RS_Model(rs_model, x_data, y_data, x_train, y_train)

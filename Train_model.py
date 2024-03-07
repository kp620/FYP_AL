import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class train_model(nn.Module):
    def __init__(self):
        super(train_model, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm1d(32)
        # self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # self.fc = nn.Linear(in_features=512 * 6, out_features=1)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32) # 32 * 38

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128) # 128 * 18
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0) # 128 * 9

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(512) # 256 * 4

        self.conv4 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(1024) # 512 * 2
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0) # 512 * 1

        self.fc = nn.Linear(in_features=1024 * 1, out_features=1)

    def part1(self, x):
        # x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 32 * 38
        # x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 128 * 18
        # x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 256 * 8
        # x = self.bn4(self.conv4(x)) # 512 * 6
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool2(x)
        return x
    
    def part2(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        if(torch.isnan(x).any()):
          print("nan here!")
        return x

    def forward_to_penultimate(self, x):
        # Forward pass up to the penultimate layer
        x = self.part1(x)
        return x

def explain_model():
    print("Building train model...")
    model = train_model()
    print(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: ", params)
    return model

def build_model():
    model = explain_model()
    # kaiming initialization
    if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv1d):
        torch.nn.init.kaiming_uniform_(model.weight)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)
    return model

def check_accuracy(model, val_loader, device, dtype, criterion):
    model.eval()  # set model to evaluation mode
    validation_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    # Disable gradient calculations
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, dtype=dtype)
            targets = targets.to(device, dtype=dtype)
            # Forward pass to get outputs
            scores = model(inputs)
            # Calculate the loss
            loss = criterion(scores, targets)
            validation_loss += loss.item()
            # Convert outputs probabilities to predicted class (1 if output > 0.5 for binary classification)
            preds = scores > 0.5
            # Count correct predictions
            correct_predictions += torch.sum(preds == targets.data).item()
            total_predictions += targets.size(0)
    # Calculate average loss and accuracy
    validation_loss /= len(val_loader)
    validation_accuracy = correct_predictions / total_predictions
    print("correct predictions: ", correct_predictions)
    print("total_predictions: ", total_predictions)
    # Return results
    return validation_loss, validation_accuracy

def train_epoch(model, train_loader, val_loader, device, dtype, criterion, learning_rate):
    model = model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for t,(x,y) in enumerate(train_loader):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            optimizer.zero_grad()
            scores = model(x)
            loss = criterion(scores,y)
            loss.backward()
            optimizer.step()
            if t % 1000 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        val_loss, val_accuracy = check_accuracy(model, val_loader, device, dtype, criterion)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}')

def gradient_train_epoch(model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion):
    gradients = []
    model = model.to(device=device)
    model.train()  # Ensure the model is in training mode

    for index, (x, y) in enumerate(unlabel_loader):
        x = x.to(device=device, dtype=dtype)
        pseudo_y = pseudo_labels[index * batch_size: index * batch_size + batch_size].to(device=device, dtype=dtype).unsqueeze(1)

        # Process input through the model up to the penultimate layer
        penultimate_output = model.forward_to_penultimate(x)

        # Detach the output of the penultimate layer from the computation graph
        penultimate_output_detached = penultimate_output.detach()
        penultimate_output_detached.requires_grad = True  # Enable gradient computation for this tensor

        scores = model.part2(penultimate_output_detached)
        loss = criterion(scores, pseudo_y)
        loss.backward()

        if penultimate_output_detached.grad is not None:
            # Convert the gradient to a NumPy array and store it
            gradients.append(penultimate_output_detached.grad.cpu().detach().numpy())
        else:
            gradients.append(None)

    gradients = np.concatenate(gradients, axis=0)
    # np.savetxt('gradients.txt', gradients.reshape(gradients.shape[0], -1))
    return gradients
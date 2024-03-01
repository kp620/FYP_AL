import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class one_dimensional_CNN(nn.Module):
    def __init__(self):
        super(one_dimensional_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(in_features=512 * 6, out_features=1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 32 * 38
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 128 * 18
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # 256 * 8
        x = self.bn4(self.conv4(x)) # 512 * 6
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        if(torch.isnan(x).any()):
          print("nan here!")
        return x


def explain_model():
    model = one_dimensional_CNN()
    print(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: ", params)
    return model

def kaiming_ini(model):
    if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv1d):
        torch.nn.init.kaiming_uniform_(model.weight)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)
    
def build_model():
    model = explain_model()
    model.apply(kaiming_ini)
    return model

def check_accuracy(model, val_loader, device, dtype, criterion, analysis=False):
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

def train_epoch(model, train_loader, val_loader, device, dtype, criterion):
    model = model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
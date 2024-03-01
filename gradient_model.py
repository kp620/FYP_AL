import torch.nn as nn
import torch.nn.functional as F
import torch

class gradient_one_dimensional_CNN(nn.Module):
    def __init__(self):
        super(gradient_one_dimensional_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32) # 32 * 38

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128) # 128 * 18
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0) # 128 * 9

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256) # 256 * 4

        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm1d(512) # 512 * 2
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0) # 512 * 1

        self.fc = nn.Linear(in_features=512 * 1, out_features=2)

    def part1(self, x):
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
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

    def part2(self, x):
        # x = F.softmax(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        return x

    def forward_to_penultimate(self, x):
        # Forward pass up to the penultimate layer
        x = self.part1(x)
        return x
    

def explain_model():
    model = gradient_one_dimensional_CNN()
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
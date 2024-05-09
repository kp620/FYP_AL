import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        pre_linear = torch.flatten(out, 1)
        out = self.linear(pre_linear)
        return out, pre_linear

def ResNet18_1D(num_classes=168):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes)

def explain_model(class_type):
    if class_type == "binary":
        print("Building binary model...")
        model = ResNet18_1D(num_classes=2)
    elif class_type == "multi":
        print("Building multiclass model...")
        model = ResNet18_1D(num_classes=168)
    # print(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: ", params)
    return model

def build_model(class_type):
    model = explain_model(class_type)
    # kaiming initialization
    if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.Conv1d):
        torch.nn.init.kaiming_uniform_(model.weight)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)
    return model

def train(model, train_loader, device, dtype, criterion, learning_rate):
    model.train()
    model = model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 50
    for epoch in range(num_epochs):
        for batch,(input, target, idx) in enumerate(train_loader):
            model.train()
            input = input.to(device=device, dtype=dtype)
            target = target.to(device=device, dtype=dtype).squeeze().long()
            optimizer.zero_grad()
            output, _ = model(input)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Epoch: ", epoch, "Loss: ", loss.item())

def gradient_train(model, unlabel_loader, pseudo_labels, device, dtype, batch_size, criterion):
    gradients = []
    model = model.to(device=device)
    model.eval()  # Ensure the model is in training mode
    for batch, (input, target, idx) in enumerate(unlabel_loader):
        input = input.to(device=device, dtype=dtype)
        pseudo_y = pseudo_labels[idx].to(device=device, dtype=dtype).squeeze().long()
        # pseudo_y = pseudo_labels[batch * batch_size: (batch + 1) * batch_size].to(device=device, dtype=dtype).squeeze().long()
        output, pre_linear_output = model(input)
        # Enable gradient retention for non-leaf tensor
        pre_linear_output.retain_grad()
        loss = criterion(output, pseudo_y)
        # Backward pass to compute gradients
        loss.backward()
        if pre_linear_output.grad is not None:
            gradients.append(pre_linear_output.grad.cpu().detach().numpy())
        else:
            gradients.append(None)
    gradients = np.concatenate(gradients, axis=0)
    model.train()
    return gradients

def psuedo_labeling(model, devc, dtype, loader):
    pseudo_labels = []
    with torch.no_grad():
        for input, target, idx in loader:
            input = input.to(devc, dtype=dtype)
            output, _ = model(input)
            probabilities = F.softmax(output, dim=1)
            _, pseudo_label = torch.max(probabilities, dim=1)
            pseudo_labels.append(pseudo_label.cpu())
    pseudo_labels = torch.cat(pseudo_labels)
    return pseudo_labels
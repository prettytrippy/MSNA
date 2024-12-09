import torch
import torch.nn as nn
import torch.nn.functional as F

class MSNA_CNN(nn.Module):
    def __init__(self, n):
        super(MSNA_CNN, self).__init__()
        self.n = n

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, dilation=1, padding=3//2)
        self.bn1 = nn.BatchNorm1d(4)
        
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=1, dilation=1, padding=5//2)
        self.bn2 = nn.BatchNorm1d(8)

        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=1, dilation=1, padding=7//2)
        self.bn3 = nn.BatchNorm1d(16)

        self.conv4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1, dilation=1, padding=9//2)
        self.bn4 = nn.BatchNorm1d(32)

        self.conv5 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=1, dilation=1, padding=11//2)
        self.bn5 = nn.BatchNorm1d(64)

        self.conv6 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=13, stride=1, dilation=1, padding=13//2)
        self.bn6 = nn.BatchNorm1d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(n*2, n)  # Adjust input size after pooling
        self.fc2 = nn.Linear(n, 1)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Final layer with sigmoid activation
        x = self.fc2(x)
        return torch.sigmoid(x)
        
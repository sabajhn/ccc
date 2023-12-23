import torch
import torch.nn as nn
import torch.nn.functional as F

# 1,6,16,12
# Creating the CNN model
class CNN_rout(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(11, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
        self.conv4 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
        # self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8, 100)

        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
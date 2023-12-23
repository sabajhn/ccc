import torch
import torch.nn as nn
import torch.nn.functional as F

# 1,6,16,12
# class CNN_ri(nn.Module):
#     def __init__(self):
#         super(CNN_ri, self).__init__()
#         # Convolutional layer 1: input 1 channel, output 16 channels, kernel size 3, stride 1, padding 1
#         self.conv1 = nn.Conv2d(11, 16, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         # Convolutional layer 2: input 16 channels, output 32 channels, kernel size 3, stride 1, padding 1
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#         # Max pooling layer: kernel size 2, stride 2
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(32,64)
#         # Fully connected layer 1: input 7*7*32 = 1568 features, output 64 features
#         # self.fc1 = nn.Linear(149248, 64)
#         # Fully connected layer 2: input 64 features, output 1 feature
#         self.fc2 = nn.Linear(64, 1)

#     def forward(self, x):
#         # Apply the first convolutional layer and activation function
#         x = self.pool(F.relu(self.conv1(x)))
#         # Apply the second convolutional layer and activation function
#         x =self.pool(F.relu(self.conv2(x)))
#         # Apply the max pooling layer
#         # x = self.pool(x)
#         # Flatten the output of the pooling layer
#         # x = x.view(-1, 18656)
#         # Apply the first fully connected layer and activation function
#         x = F.relu(self.fc1(x))
#         # Apply the second fully connected layer and activation function
#         x = torch.sigmoid(self.fc2(x))
#         # Return the output
#         return x

# # Creating the CNN model
class CNN_ri(nn.Module):
    def __init__(self):
        super(CNN_ri, self).__init__()
        self.conv1 = nn.Conv2d(11, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
        # self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320,100)
        # self.fc2 = nn.Linear(80, 100)

        # self.fc3 = nn.Linear(100, 100)

        # self.fc4 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.pool(F.relu(self.conv4(x)))
        # x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = self.fc3(x)
        x = self.sigmoid(x)
        return x
# class cnn_model(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(6,3,5,stride=(2,1))
#         self.pool = nn.MaxPool2d(3, 2,stride=(3,2))
#         self.conv2 = nn.Conv2d(3, 3, 5,stride=(2,2))
#         self.pool = nn.MaxPool2d(2, 2,stride=(2,2))
#         self.conv3 = nn.Conv2d(3, 3, 5,stride=(1,1))
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.nn.AvgPool2d(5, stride=2)
#         # GovalAvgPooling()
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


    #     self.conv1 = nn.Conv2d(6,32,5,stride=(2,1))
    #     self.pool = nn.MaxPool2d(2, 2,stride=(2,1))
    #     self.conv2 = nn.Conv2d(6, 16, 5,stride=(2,1))

    #     self.conv2 = nn.Conv2d(6, 16, 5,stride=(2,1))
    #     # self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     # self.fc2 = nn.Linear(120, 84)
    #     # self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     # x = torch.flatten(x, 1) # flatten all dimensions except batch
    #     # x = F.relu(self.fc1(x))
    #     # x = F.relu(self.fc2(x))
    #     # x = self.fc3(x)
    #     return x

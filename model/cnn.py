import torch.nn as nn
import torch.nn.functional as F

# 1,6,16,12
# Creating the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layer with 32 filters and 3x3 kernel size, using same padding to preserve the input size
        self.conv1 = nn.Conv2d(10, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
        self.conv4 = nn.Conv2d(16, 8, 3, padding=1)
        # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
        self.conv5 = nn.Conv2d(8, 2, 3, padding=1)

    def forward(self, x):
        # Applying the convolutional layers with ReLU activation function
        # x = self.conv1(x)
        # x = self.conv1(x)
        # x = self.conv1(x)
        # x = self.conv1(x)
        x = F.relu(self.conv1(x))
        x= F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x= F.relu(self.conv5(x))
        # x = torch.relu(self.conv1(x))
        # x = torch.relu(self.conv2(x))
        # x = torch.relu(self.conv3(x))
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

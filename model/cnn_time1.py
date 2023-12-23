import torch
import torch.nn as nn
import torch.nn.functional as F
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.cancel import CancelOut

class CNN_croute2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(11, 64, (1,1))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, (1,1))
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(32, 16, (1,1))
        # #  be multiplied (16x592 and 62x400)
        # self.pool = nn.MaxPool2d(2, 2)
        # # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
        # self.conv4 = nn.Conv2d(16, 8, (1,1))
        # self.pool = nn.MaxPool2d(2, 2)
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x586496 and 1314048x120)
        # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
        # self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(592, 400)
        self.fc1 = nn.Linear(21120, 400)
        # self.fc3 = nn.Linear(400, 400)

        self.fc4 = nn.Linear(400, 100)
        self.fc5 = nn.Linear(100, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x,gf):
        x = x.float()
        # print(x.shape," a")
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape," b")
        x = self.pool(F.relu(self.conv2(x)))
        # # print(x.shape," c")
        # x = self.pool(F.relu(self.conv3(x)))
        # # print(x.shape," b")
        # x = self.pool(F.relu(self.conv4(x)))

        # x = self.pool(self.conv3(x))
        # x = self.pool(self.conv4(x))
        # x = self.pool(F.relu(self.conv3(x)))
        # print(x.shape," e")/
        # x = self.pool(F.relu(self.conv4(x)))
        # print(x.shape," f")
        # x = F.adaptive_avg_pool2d(x,(1,1))

        x = torch.flatten(torch.tensor(x), 1)
        # x = torch.concat([x,gf],1)
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
# # Define the CNN model class
# class CNN_t(nn.Module):
#     def __init__(self):
#         super(CNN_t, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(11, 16, 3, padding=1) # input: 1 channel, output: 16 channels, kernel size: 3x3
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # input: 16 channels, output: 32 channels, kernel size: 3x3
#         self.pool = nn.MaxPool2d(2, 2)
#         # self.conv3 = nn.Conv2d(32, 64, (6,7), padding=1) # input: 32 channels, output: 64 channels, kernel size: 3x3
#         # Pooling layer
#         # self.pool = nn.MaxPool2d(2, 2) # kernel size: 2x2, stride: 2
#         self.conv3 = nn.Conv2d(32,64, 5, padding=1) # input: 32 channels, output: 64 channels, kernel size: 3x3
#         # Pooling layer
#         self.pool = nn.MaxPool2d(2, 2)
#         # Fully connected layers
#         self.fc1 = nn.Linear(128, 256) # input: 64 * 4 * 4 features, output: 256 features
#         self.fc2 = nn.Linear(256, 148)
#         self.fc3 = nn.Linear(148, 48) # input: 256 features, output: 48 features
#         self.fc4 = nn.Linear(48, 1) # input: 48 features, output: 1 value

#     def forward(self, x):
#         # Apply convolutional and pooling layers
#         x = self.pool(F.relu(self.conv1(x))) # shape: (batch_size, 16, 14, 14)
#         x = self.pool(F.relu(self.conv2(x))) # shape: (batch_size, 32, 7, 7)
#         x = self.pool(F.relu(self.conv3(x))) # shape: (batch_size, 64, 4, 4)
#         # x = self.pool(F.relu(self.conv4(x)))
#         # Flatten the output
#         x = torch.flatten(x, 1)
#         # x = x.view(-1, 64 * 3 * 3) # shape: (batch_size, 64 * 4 * 4)
#         # Apply fully connected layers and activation functions
#         x = F.relu(self.fc1(x)) # shape: (batch_size, 256)
#         x = F.relu(self.fc2(x)) # shape: (batch_size, 48)
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x) # shape: (batch_size, 1)
#         return x










# 1,6,16,12
# Creating the CNN model
# class CNN_t(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(11, 64, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(64,32, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         # self.conv3 = nn.Conv2d(32,16, 3)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
#         # self.conv4 = nn.Conv2d(16, 8, 3)
#         # self.pool = nn.MaxPool2d(2, 2)
        
#         # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
#         # self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
#         # self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(384, 100)

#         self.fc2 = nn.Linear(100, 100)
#         self.fc3 = nn.Linear(100, 1)
#         # self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         # x = self.pool(F.relu(self.conv3(x)))
#         # x = self.pool(F.relu(self.conv4(x)))
#         # print(x.shape)
#         # print(gf.shape)
#         # all_features = torch.concat([x, gf.view(1,-1)],1) 
#         # x=all_features
#         # x = self.pool(F.relu(self.conv4(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         # x = self.sigmoid(x)
#         return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# 1,6,16,12
# Creating the CNN model
# class CNN_t(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(11, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
#         self.conv4 = nn.Conv2d(16, 8, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
        
#         # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
#         # self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
#         # self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(8, 100)

#         self.fc2 = nn.Linear(100, 100)
#         self.fc3 = nn.Linear(100, 1)
#         # self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.pool(F.relu(self.conv4(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         # x = self.sigmoid(x)
#         return x
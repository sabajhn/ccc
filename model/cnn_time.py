import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.cancel import CancelOut


class CNN_croute(nn.Module):
    def __init__(self):
        super().__init__()
        # self.cancelout = CancelOut(16)
        self.conv1 = nn.Conv2d(11, 64, (3,3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, (3,2))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 16, (2,1))
        # #  be multiplied (16x592 and 62x400)
        self.pool = nn.MaxPool2d(2, 2)
        # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
        # self.conv4 = nn.Conv2d(16, 8, (2,2))
        # self.pool = nn.MaxPool2d(2, 2)
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x586496 and 1314048x120)
        # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
        # self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(592, 400)
        self.fc1 = nn.Linear(926, 400)

        self.fc2 = nn.Linear(400, 100)
        # self.fc3 = nn.Linear(100, 100)
        # self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x,gf):
        x = x.float()
        # print(x.shape,"x sh1")
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(torch.tensor(x), 1)
        # print(x.shape,"x sh")
        # print(gf.shape, "gh sh")
        x = torch.concat([x.float(),gf.float()],1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x


# Creating the CNN model
# class CNN_croute(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.cancelout = CancelOut(16)
#         self.conv1 = nn.Conv2d(11, 64, (3,3))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(64, 32, (3,3))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(32, 16, (3,3))
#         #  be multiplied (16x592 and 62x400)
#         self.pool = nn.MaxPool2d(2, 2)
#         # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
#         # self.conv4 = nn.Conv2d(16, 8, (2,2))
#         # self.pool = nn.MaxPool2d(2, 2)
#         # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x586496 and 1314048x120)
#         # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
#         # self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.fc1 = nn.Linear(592, 400)
#         self.fc1 = nn.Linear(671, 400)

#         self.fc2 = nn.Linear(400, 100)
#         self.fc3 = nn.Linear(100, 100)
#         self.fc4 = nn.Linear(100, 1)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x,gf):
#         x = x.float()
#         # print(x.shape," a")
#         # x = self.cancelout(x)
#         x = self.pool(F.relu(self.conv1(x)))
#         # print(x.shape," b")
#         x = self.pool(F.relu(self.conv2(x)))
#         # print(x.shape," c")
#         x = self.pool(F.relu(self.conv3(x)))

#         # x = F.relu(self.conv3(x))
#         # x = self.pool(F.relu(self.conv3(x)))
#         # print(x.shape," e")/
#         # x = self.pool(F.relu(self.conv4(x)))
#         # x = self.pool(F.relu(self.conv5(x)))
#         # print(x.shape," f")
#         # x = F.adaptive_avg_pool2d(x,(1,1))

#         x = torch.flatten(torch.tensor(x), 1)
#         x = torch.concat([x.float(),gf.float()],1)
#         # x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))

#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         x = self.sigmoid(x)
#         return x


# class CNN_croute(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.cancelout = CancelOut(16)
#         self.conv1 = nn.Conv2d(11, 64, (3,3))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(64, 32, (3,3))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(32, 16, (3,3))
#         #  be multiplied (16x592 and 62x400)
#         self.pool = nn.MaxPool2d(2, 2)
#         # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
#         self.conv4 = nn.Conv2d(16, 8, (2,2))
#         self.pool = nn.MaxPool2d(2, 2)
#         # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x586496 and 1314048x120)
#         # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
#         self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         # self.fc1 = nn.Linear(592, 400)
#         self.fc1 = nn.Linear(33, 400)

#         self.fc2 = nn.Linear(400, 100)
#         self.fc3 = nn.Linear(100, 1)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x,gf):
#         x = x.float()
#         # print(x.shape," a")
#         # x = self.cancelout(x)
#         x = self.pool(F.relu(self.conv1(x)))
#         # print(x.shape," b")
#         x = self.pool(F.relu(self.conv2(x)))
#         # print(x.shape," c")
#         x = self.pool(F.relu(self.conv3(x)))

#         # x = F.relu(self.conv3(x))
#         # x = self.pool(F.relu(self.conv3(x)))
#         # print(x.shape," e")/
#         x = self.pool(F.relu(self.conv4(x)))
#         x = self.pool(F.relu(self.conv5(x)))
#         # print(x.shape," f")
#         # x = F.adaptive_avg_pool2d(x,(1,1))

#         x = torch.flatten(torch.tensor(x), 1)
#         x = torch.concat([x,gf],1)
#         # x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x





# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from models.cancel import CancelOut

# # Creating the CNN model
# class CNN_croute(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.cancelout = CancelOut(16)
#         self.conv1 = nn.Conv2d(11, 64, (3,3))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(64, 32, (3,3))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(32, 16, (3,3))
#         #  be multiplied (16x592 and 62x400)
#         self.pool = nn.MaxPool2d(2, 2)
#         # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
#         self.conv4 = nn.Conv2d(16, 8, (2,2))
#         self.pool = nn.MaxPool2d(2, 2)
#         # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x586496 and 1314048x120)
#         # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
#         self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         # self.fc1 = nn.Linear(592, 400)
#         self.fc1 = nn.Linear(34, 400)

#         self.fc2 = nn.Linear(400, 100)
#         self.fc3 = nn.Linear(100, 1)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x,gf):
#         x = x.float()
#         # print(x.shape," a")
#         # x = self.cancelout(x)
#         x = self.pool(F.relu(self.conv1(x)))
#         # print(x.shape," b")
#         x = self.pool(F.relu(self.conv2(x)))
#         # print(x.shape," c")
#         x = self.pool(F.relu(self.conv3(x)))

#         # x = F.relu(self.conv3(x))
#         # x = self.pool(F.relu(self.conv3(x)))
#         # print(x.shape," e")/
#         x = self.pool(F.relu(self.conv4(x)))
#         x = self.pool(F.relu(self.conv5(x)))
#         # print(x.shape," f")
#         # x = F.adaptive_avg_pool2d(x,(1,1))

#         x = torch.flatten(torch.tensor(x), 1)
#         x = torch.concat([x,gf],1)
#         # x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x




























# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # # from models.cancel import CancelOut

# # # Creating the CNN model
# # class CNN_croute(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         # self.cancelout = CancelOut(16)
# #         self.conv1 = nn.Conv2d(11, 64, (1,1))
# #         self.pool = nn.MaxPool2d(2, 2)
# #         self.conv2 = nn.Conv2d(64, 32, (1,1))
# #         self.pool = nn.MaxPool2d(2, 2)
# #         self.conv3 = nn.Conv2d(32, 16, (1,1))
# #         #  be multiplied (16x592 and 62x400)
# #         self.pool = nn.MaxPool2d(2, 2)
# #         # Convolutional layer with 64 filters and 3x3 kernel size, using same padding to preserve the input size
# #         self.conv4 = nn.Conv2d(16, 8, (1,1))
# #         self.pool = nn.MaxPool2d(2, 2)
# #         # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x586496 and 1314048x120)
# #         # Convolutional layer with 3 filters and 3x3 kernel size, using same padding to preserve the input size
# #         self.conv5 = nn.Conv2d(8, 2, 3, padding=1)
# #         self.pool = nn.MaxPool2d(2, 2)
# #         # self.fc1 = nn.Linear(592, 400)
# #         self.fc1 = nn.Linear(36, 400)
# #         # 44

# #         self.fc2 = nn.Linear(400, 100)
# #         self.fc3 = nn.Linear(100, 1)
# #         self.sigmoid = torch.nn.Sigmoid()

# #     def forward(self, x,gf):
# #         x = x.float()
# #         print(x.shape," a")
# #         # x = self.cancelout(x)
# #         x = self.pool(F.relu(self.conv1(x)))
# #         # print(x.shape," b")
# #         x = self.pool(F.relu(self.conv2(x)))
# #         # print(x.shape," c")
# #         x = self.pool(F.relu(self.conv3(x)))

# #         # x = F.relu(self.conv3(x))
# #         # x = self.pool(F.relu(self.conv3(x)))
# #         # print(x.shape," e")/
# #         x = self.pool(F.relu(self.conv4(x)))
# #         x = self.pool(F.relu(self.conv5(x)))
# #         # print(x.shape," f")
# #         # x = F.adaptive_avg_pool2d(x,(1,1))

# #         x = torch.flatten(torch.tensor(x), 1)
# #         print("SHAPE", x.shape)
# #         print( gf.shape)
# #         x = torch.concat([x,gf],1)
# #         # x = torch.flatten(x, 1) # flatten all dimensions except batch
# #         x = F.relu(self.fc1(x))
# #         x = F.relu(self.fc2(x))
# #         x = self.fc3(x)
# #         x = self.sigmoid(x)
# #         return x



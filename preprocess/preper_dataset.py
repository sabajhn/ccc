import os
import torch
from torch.utils.data import Dataset
# import pandas as pd
# from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, data_input, data_ouput , transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.data_ouput = data_ouput
        self.data_input = data_input

    def __len__(self):
        return len(self.data_ouput)

    def __getitem__(self, idx):
        label_ = self.data_ouput[idx]
        input_ = self.data_input[idx]
        return input_.float(),label_.float()
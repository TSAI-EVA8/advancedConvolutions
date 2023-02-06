import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class PyTorchImageDataset(Dataset):
    def __init__(self, CIFAR_data, datasetType="Train",transformations=None):
        self.CIFAR_data = CIFAR_data
        self.transformations = transformations
        self.datasetType=datasetType

    
    def __getitem__(self, index):
        image, label = self.CIFAR_data[index]
        if(self.datasetType=="Train" and self.transformations is not None):
          image = self.transformations(image=np.array(image))['image']
          image = np.transpose(image, (2, 0, 1)).astype(np.float32)
          #image = self.transformations(image)
          #print(image)
          #image = image.astype(np.float32)
          # Returning -> Inputs, Labels
          # Data -> ImageData, RandomNumberVector
          # Labels -> ImageLabel, RandomNumber 
          return torch.tensor(image, dtype=torch.float),label

        elif(self.datasetType=="Test" and self.transformations is not None):
          image = self.transformations(image)
          return image,label

    def __len__(self):
        return len(self.CIFAR_data)


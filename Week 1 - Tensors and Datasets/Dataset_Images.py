import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as dsets
from PIL import Image
import pandas as pd
# import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(__file__)
base = os.path.join(script_dir,"..","Dataset")

MNIST_dataset = dsets.MNIST(root=base, download= True, transform= transforms.ToTensor())

directory = os.path.join(base,"DLab")
csv_file = "index.csv"
csv_path = os.path.join(directory,csv_file)

data_name = pd.read_csv(csv_path)
data_name.head()

print('File name :', data_name.iloc[1,1])
print('class or y :', data_name.iloc[1,0])

image_name = data_name.iloc[1,1]
image_path = os.path.join(directory, image_name)

image = Image.open(image_path)
# plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)
# plt.title(data_name.iloc[1,0])
# plt.show()

class Dataset(Dataset):
    
    def __init__(self, csv_path, data_dir, transform = None):

        self.transform = transform
        self.data_dir = data_dir
        self.data_name = pd.read_csv(csv_path)
        self.len = data_name.shape[0]
    
    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        
        img_name = self.data_name.iloc[idx, 1]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path)

        y = self.data_name.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)
        return image,y

transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])

# dataset = Dataset(csv_path, directory, transform)

# print(dataset[0][0].shape)


dataset = MNIST_dataset
print("Type of the first element: ", type(dataset[0]))
print("The length of the tuple: ", len(dataset[0]))
print("The shape of the first element in the tuple: ", dataset[0][0].shape)
print("The type of the first element in the tuple", type(dataset[0][0]))
print("The second element in the tuple: ", dataset[0][1])
print("The type of the second element in the tuple: ", type(dataset[0][1]))
print("As the result, the structure of the first element in the dataset is (tensor([1, 28, 28]), tensor(7)).")

# Combine two transforms: crop and convert to tensor. Apply the compose to MNIST dataset

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root=base, download= False, transform = croptensor_data_transform)
print("The shape of the first element in the first tuple: ", dataset[0][0].shape)
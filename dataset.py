import torch
import torchvision
import torchvision.transforms as transforms
import glob
import os

import torch.nn as nn
import torchvision.datasets as dsets
import pandas as pd
from torchvision.io import read_image
import numpy as np


class CustomDatasetFromImages():
    def __init__(self,img_dir, csv_path,transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        #image Dir
        self.img_dir = img_dir
        # Read the csv file
        self.df = pd.read_csv(csv_path)
        # First column contains the image paths and image name
        self.image_arr = self.df.iloc[:, 0]
        # forth column is for an label
        self.label = self.df.iloc[:, 3]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.label)


    def __getitem__(self,idx):
        
        img_path = os.path.join(self.img_dir, self.image_arr[idx])
        images = read_image(img_path)
        # images = images.permute((2, 0, 1))      # convert to (C, H, W) format
        label = self.label[idx]
        if self.transforms is not None:
            images = self.transforms(images)

        return images , label


batch_size=1
img_dir ='images'
annotations_file = "train.csv"
input_size = 224



transforms = transforms.Compose([transforms.ToPILImage(),transforms.Resize([int(224), int(224)]),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = CustomDatasetFromImages(img_dir,annotations_file,transforms=transforms )
img_dir ='images'
annotations_file = "val.csv"

val_dataset = CustomDatasetFromImages(img_dir,annotations_file ,transforms=transforms)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset ,batch_size=batch_size, shuffle=False, pin_memory=True)

# for idx ,(images ,label) in enumerate(val_loader):
    # print('image_size',images.size(),"label_size",label.size())
    # print(label)
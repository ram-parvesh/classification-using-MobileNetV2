import torch
import torchvision
import torchvision.transforms as transforms
import glob
import os
import datetime,time
import torch.nn as nn
import torchvision.datasets as dsets
import pandas as pd
from torchvision.io import read_image
import numpy as np

from dataset import CustomDatasetFromImages
from MobileNet import MobileNetV2

img_dir ='images'
test_annotations_file = "test.csv"
batch_size = 1

transforms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([int(224), int(224)]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_dataset = CustomDatasetFromImages(img_dir,test_annotations_file,transforms=transforms )


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

model_dir = 'result_batch32.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --------- 3. model define ---------
net = MobileNetV2(width_mult=1).to(device)

net.load_state_dict(torch.load(model_dir))
net.eval() # eval model (batchnorm uses moving mean./varience instead of mini-batch mean and varience)
with torch.no_grad():
    correct = 0
    total = 0
    for images ,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        # print("outputs data",outputs.data)
        _ ,predicted =torch.max(outputs.data,1)
        # print('outputs:',predicted,'label:',labels)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    print('Test Accuracy of the model on the  test images: {} %'.format(100 * correct / total))
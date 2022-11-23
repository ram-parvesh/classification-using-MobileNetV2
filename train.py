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
from torch.utils.tensorboard import SummaryWriter
from dataset import CustomDatasetFromImages
from MobileNet import MobileNetV2


writer = SummaryWriter()
batch_size=32
img_dir ='images'
train_annotations_file = "train.csv"
val_annotations_file = "val.csv"

transforms = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([int(224), int(224)]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = CustomDatasetFromImages(img_dir,train_annotations_file,transforms=transforms )
val_dataset = CustomDatasetFromImages(img_dir,val_annotations_file ,transforms=transforms)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset ,batch_size=batch_size, shuffle=True, pin_memory=False)

num_epoch = 50
learning_rate = 0.0001
input_size = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = MobileNetV2(width_mult=1).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
total_step = len(train_loader)
print("Total step: ",total_step)

valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []

for epoch in range(num_epoch):
    running_loss = 0.0
    # scheduler.step(epoch)
    correct = 0
    total=0
    start_time=datetime.datetime.now()
    print("Training started at time :{} epoch :{}".format(start_time,epoch))
    for idx ,(images,label) in enumerate(train_loader):
        # print(images.size())
        images = images.to(device)
        label = label.to(device)

        outputs =net(images)
        loss = criterion(outputs,label)



        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # print statistics
        running_loss += loss.item()
        _,pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==label).item()
        total += label.size(0)
        
        if idx % 1000 == 0:
            print('Epoch [{}/{}],Step [{}/{}],loss:{:.4f}'.format(epoch+1,num_epoch,idx,total_step,loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    
    writer.add_scalar("Training Loss",(running_loss/total_step),epoch)
    writer.add_scalar("Training Accuracy",(100 * correct / total),epoch)
    print(f'Training loss: {np.mean(train_loss):.4f}, Training acc: {(100 * correct / total):.4f}\n')      

    correct_t = 0
    total_t = 0
    batch_loss = 0

    with torch.no_grad():
        net.eval()    # eval model 
        for data_t ,target_t in val_loader:
            data_t = data_t.to(device)
            target_t = target_t.to(device)
            outputs_t = net(data_t)

            # print("outputs data",outputs.data)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss/len(val_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        writer.add_scalar("validation loss",(batch_loss/len(val_loader)),epoch)
        writer.add_scalar("validation Accuracy",(100 * correct_t / total_t),epoch)
    # Save the model checkpoint
        if (epoch +1) % 2==0:
            torch.save(net.state_dict(), 'result0.000132_'+str(epoch+1)+'_model.pth')

writer.close()


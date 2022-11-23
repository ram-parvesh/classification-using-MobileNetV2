import torch
import torchvision
import torchvision.transforms as transforms
import glob
import os
import datetime,time
import torch.nn as nn
import torchvision.datasets as dsets
import pandas as pd
import math
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
val_loader = torch.utils.data.DataLoader(dataset=val_dataset ,batch_size=batch_size, shuffle=True, pin_memory=True)

num_epoch = 150
learning_rate = 0.05
input_size = 224



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = MobileNetV2(width_mult=1).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def update_lr(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
curr_lr = learning_rate

lr_end = 0.00004
def cosine_decay(epoch):
    if num_epoch > 1:
        w = (1 + math.cos(epoch / (num_epoch-1) * math.pi)) / 2
    else:
        w = 1
    return w * learning_rate + (1 - w) * lr_end

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
        
        if idx % 100 == 0:
            print('Epoch [{}/{}],Step [{}/{}],loss:{:.4f}'.format(epoch+1,num_epoch,idx,total_step,loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)

    # Decay learning rate
    curr_lr = cosine_decay(epoch)
    update_lr(optimizer, curr_lr) 
    # print("current _learning rate",curr_lr)    
    
    
    writer.add_scalar("Training Loss",(running_loss/total_step),epoch)
    writer.add_scalar("Training Accuracy",(100 * correct / total),epoch)
    print(f'Training loss: {np.mean(train_loss):.4f}, Training acc: {(100 * correct / total):.4f}\n')
    end_time =datetime.datetime.now()
    print("Training time after each  epoch :",(end_time-start_time))



    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()    # eval model (batchnorm uses moving mean./varience instead of mini-batch mean and varience)
        for data_t ,target_t in val_loader:
            data_t = data_t.to(device)
            target_t = target_t.to(device)
            outputs_t = net(data_t)

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

        # Saving the best weight 
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'model.pth')
            
        # Save the model checkpoint
        if (epoch +1) % 10==0:
            torch.save(net.state_dict(), 'epoch_'+str(epoch)+'_model.pth')

writer.close()






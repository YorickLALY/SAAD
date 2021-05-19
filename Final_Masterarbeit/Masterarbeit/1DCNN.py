# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:53:14 2021

@author: lalyor
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from Dataset import Dataset


print("Hello world")
print(torch.cuda.is_available())

######################################################################################################################

signal_silver = []
signal_black = []

for j in range(1,6):
    for i in range(16):
        # Open data from silver bucket
        if i < 8:
            # open the file in read binary mode
            file = open("C:/Users/lalyor/Documents/Masterarbeit/Runs_8/Silver"+str(j)+"/signal_silver_"+str(i), "rb")
            #read the file to numpy array
            arr1 = np.load(file)
            signal_silver += [arr1]
            #close the file
            file.close()
        else: # Open data from black bucket
            # open the file in read binary mode
            file = open("C:/Users/lalyor/Documents/Masterarbeit/Runs_8/Black"+str(j)+"/signal_black_"+str(i-8), "rb")
            #read the file to numpy array
            arr1 = np.load(file)
            signal_black += [arr1]
            #close the file
            file.close()


signal_silver = np.array(signal_silver)
signal_black = np.array(signal_black)

batch = np.concatenate((signal_silver[0:8,:],signal_black[0:8,:]),axis=1)

for i in range(1,5):
    batch = np.concatenate((batch,signal_silver[8*i:8*(i+1),:],signal_black[8*i:8*(i+1),:]),axis=1)

batch_silver = []
for i in range(0,980):
    batch_silver += [batch[:,i:(i+20)]]

######################################################################################################################

dilation_size = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(8,8,3,padding=1)
        self.conv2 = nn.Conv1d(8,8,3,padding=1)
        self.conv3 = nn.Conv1d(8,8,3,padding=1)
        self.conv4 = nn.Conv1d(8,8,3,padding=1)
        self.conv5 = nn.Conv1d(8,8,3,padding=1)
        self.conv6 = nn.Conv1d(24,8,1)
        
    def forward(self, x):   
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x_1 = F.relu(self.conv4(x))
        
        x_concat = torch.cat((x,x_1),1) 
        x_2 = F.relu(self.conv5(x_1))
        
        x_concat = torch.cat((x_concat,x_2),1)
        x = self.conv6(x_concat)
        
        return x

######################################################################################################################

input_signal_silver = batch_silver[0:960]
prediction_silver = batch_silver[20:980]
    
training_set_silver = Dataset(input_signal_silver, prediction_silver)  #Create the Dataset from the lists
trainloader_silver = torch.utils.data.DataLoader(training_set_silver, batch_size=1,shuffle=False)   #Create random batch of size 16 in order to improve the learning of the network.

#net = torch.load('CNN.pt')  #Load an existing network
net = Net()                   #Create a new network

if torch.cuda.device_count() > 1:  #If you want to use many GPUSs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

# net.to(device)

criterion = nn.MSELoss()   
# criterion_GPU = criterion.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

epoch_loss_array = []
count_epoch = 0
    
for epoch in range(50):
    
    epoch_loss = 0.0
    
    for i, data in enumerate(trainloader_silver,0):
        input_signals, signal_to_predict = data
        # input_signals_GPU = input_signals.to(device)
        # signal_to_predict_GPU = signal_to_predict.to(device)
        
        optimizer.zero_grad()
        predicted_signals = net(input_signals)
        loss = criterion(predicted_signals, signal_to_predict)
        loss.backward()
        optimizer.step()
        
        torch.save(net,'CNN.pt')
        
        running_loss = loss.item()
        epoch_loss += loss.item()
        count_epoch +=1
        
        # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        
    epoch_loss_array += [epoch_loss/count_epoch]
    

plt.plot(epoch_loss_array)     
plt.xlabel('Epoch')   
plt.ylabel('Loss')
plt.title('Evolution of the loss through the epoch for 1D-CNN')
plt.show()
   
print("End")
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:17:26 2021

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
import functions as f


print("Hello world")
print(torch.cuda.is_available())

######################################################################################################################
signal = []
window_size = 20
boolean_threshold = 1
learning_on = 1

print('window_size = ', window_size)

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Run_30_min_8/signal_"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    signal += [arr1]
    #close the file
    file.close()

signal = np.array(signal)
train_data = signal[:,0:(35000+window_size)]
test_data = signal[:,35000:(len(signal[0]))]
    
batch_train_data = f.create_batch(train_data, window_size)
batch_test_data = f.create_batch(test_data, window_size)

batch = f.create_batch(signal, window_size)

######################################################################################################################
angriff_rename_attack_1 = []
angriff_rename_attack_2 = []

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Rename_attack/signal_silver"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    angriff_rename_attack_1 += [arr1]
    #close the file
    file.close()
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Rename_attack/signal_silver_process_on"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    angriff_rename_attack_2 += [arr1]
    #close the file
    file.close()


angriff_rename_attack_1 = np.array(angriff_rename_attack_1)
angriff_rename_attack_2 = np.array(angriff_rename_attack_2)
            
batch_angriff_rename_attack_1 = f.create_batch(angriff_rename_attack_1, window_size)
batch_angriff_rename_attack_2 = f.create_batch(angriff_rename_attack_2, window_size)

pred_rename_attack_1 = [1 for i in range(990)] #Replay attack
for i  in range(113,990):
    pred_rename_attack_1[i] = 0
    
pred_rename_attack_2 = [1 for i in range(990)] #Replay attack
for i  in range(267,990):
    pred_rename_attack_2[i] = 0
    
batch_pred_rename_attack_1 = f.create_batch(pred_rename_attack_1, window_size, pred = 1)
batch_pred_rename_attack_2 = f.create_batch(pred_rename_attack_2, window_size, pred = 1)
batch_pred_rename_attack_1 = np.array(batch_pred_rename_attack_1)
batch_pred_rename_attack_2 = np.array(batch_pred_rename_attack_2)

######################################################################################################################
angriff_replay_attack_1 = []
angriff_replay_attack_2 = []

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Replay_attack/signal_silver"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    angriff_replay_attack_1 += [arr1]
    #close the file
    file.close()
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Replay_attack/signal_silver_process_on"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    angriff_replay_attack_2 += [arr1]
    #close the file
    file.close()

angriff_replay_attack_1 = np.array(angriff_replay_attack_1)
angriff_replay_attack_2 = np.array(angriff_replay_attack_2)
            
batch_angriff_replay_attack_1 = f.create_batch(angriff_replay_attack_1, window_size)
batch_angriff_replay_attack_2 = f.create_batch(angriff_replay_attack_2, window_size)

pred_replay_attack_1 = [1 for i in range(990)] #Replay attack
for i  in range(395,990):
    pred_replay_attack_1[i] = 0
    
pred_replay_attack_2 = [1 for i in range(990)] #Replay attack
for i  in range(476,990):
    pred_replay_attack_2[i] = 0
    
batch_pred_replay_attack_1 = f.create_batch(pred_replay_attack_1, window_size, pred = 1)
batch_pred_replay_attack_2 = f.create_batch(pred_replay_attack_2, window_size, pred = 1)
batch_pred_replay_attack_1 = np.array(batch_pred_replay_attack_1)
batch_pred_replay_attack_2 = np.array(batch_pred_replay_attack_2)

######################################################################################################################
angriff_sis_attack_1 = []
angriff_sis_attack_2 = []

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Sis_attack/signal_silver"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    angriff_sis_attack_1 += [arr1]
    #close the file
    file.close()
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Sis_attack/signal_silver_process_on"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    angriff_sis_attack_2 += [arr1]
    #close the file
    file.close()

angriff_sis_attack_1 = np.array(angriff_sis_attack_1)
angriff_sis_attack_2 = np.array(angriff_sis_attack_2)
            
batch_angriff_sis_attack_1 = f.create_batch(angriff_sis_attack_1, window_size)
batch_angriff_sis_attack_2 = f.create_batch(angriff_sis_attack_2, window_size)

pred_sis_attack = [0 for i in range(len(angriff_sis_attack_1[0]))] #no change in the behaviour of the process
pred_sis_attack = np.array(pred_sis_attack)
batch_pred_sis_attack = f.create_batch(pred_sis_attack, window_size, pred = 1)
batch_pred_sis_attack = np.array(batch_pred_sis_attack)

######################################################################################################################
angriff_fake_attack_1 = []
angriff_fake_attack_2 = []

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Silver/signal_silver_"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    angriff_fake_attack_1 += [arr1]
    #close the file
    file.close()
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Black/signal_black_"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    angriff_fake_attack_2 += [arr1]
    #close the file
    file.close()

angriff_fake_attack_1 = np.array(angriff_fake_attack_1)
angriff_fake_attack_2 = np.array(angriff_fake_attack_2)
            
batch_angriff_fake_attack_1 = f.create_batch(angriff_fake_attack_1, window_size)
batch_angriff_fake_attack_2 = f.create_batch(angriff_fake_attack_2, window_size)

pred_fake_attack = [1 for i in range(len(angriff_fake_attack_1[0]))] #Attack when Schranke goes down
pred_fake_attack = np.array(pred_fake_attack)
for i  in range(21*3,46*3):
    pred_fake_attack[i] = 0
    
batch_pred_fake_attack = f.create_batch(pred_fake_attack, window_size, pred = 1)
batch_pred_fake_attack = np.array(batch_pred_fake_attack)

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
if learning_on == 1:
    input_signal = batch_train_data[0:(len(batch_train_data)-window_size)]
    prediction_signal = batch_train_data[window_size:len(batch_train_data)]
        
    training_set = Dataset(input_signal, prediction_signal)  #Create the Dataset from the lists
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=1,shuffle=False)   #Create random batch of size 16 in order to improve the learning of the network.
    
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
        
    for epoch in range(10):
        
        epoch_loss = 0.0
        
        print("epoch: ", epoch)
        
        for i, data in enumerate(trainloader,0):
            input_signals, signal_to_predict = data
            # input_signals_GPU = input_signals.to(device)
            # signal_to_predict_GPU = signal_to_predict.to(device)
            
            optimizer.zero_grad()
            predicted_signals = net(input_signals)
            loss = criterion(predicted_signals, signal_to_predict)
            loss.backward()
            optimizer.step()
            
            torch.save(net,'10_ep_CNN_window_size_'+str(window_size)+'.pt')
            
            running_loss = loss.item()
            epoch_loss += loss.item()
            count_epoch +=1
            
            # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            
        epoch_loss_array += [epoch_loss/count_epoch]
        
    
    plt.plot(epoch_loss_array)     
    plt.xlabel('Epoch')   
    plt.ylabel('Loss')
    plt.title('Evolution of the loss through the epoch for 1D-CNN window size = ' + str(window_size))
    plt.show()
else:
    net = torch.load('CNN_window_size_'+str(window_size)+'.pt')  #Load an existing network

######################################################################################################################
threshold_01 = 0.001
threshold_1 = 0.01
threshold_10 = 0.1
threshold_25 = 0.25
threshold_50 = 0.5
threshold_75 = 0.75
threshold_90 = 0.9
threshold_95 = 0.95
threshold_99 = 0.99
threshold_150 = 1.50
threshold_200 = 2.00
threshold_250 = 2.50
threshold_300 = 3.00
threshold_320 = 3.20

threshold = threshold_200

test_batch = batch_test_data

pred_test = [1 for i in range(len(signal[0]))]
    
batch_pred_test = f.create_batch(pred_test, window_size, pred = 1)
batch_pred_test = np.array(batch_pred_test)

excel_result = f.evaluate_1DCNN(test_batch, batch_pred_test, net, threshold, window_size)
######################################################################################################################
excel_result += f.evaluate_1DCNN(batch_angriff_rename_attack_1, batch_pred_rename_attack_1, net, threshold, window_size)

excel_result += f.evaluate_1DCNN(batch_angriff_rename_attack_2, batch_pred_rename_attack_2, net, threshold, window_size)

######################################################################################################################
excel_result += f.evaluate_1DCNN(batch_angriff_replay_attack_1, batch_pred_replay_attack_1, net, threshold, window_size)

excel_result += f.evaluate_1DCNN(batch_angriff_replay_attack_2, batch_pred_replay_attack_2, net, threshold, window_size)

######################################################################################################################
excel_result += f.evaluate_1DCNN(batch_angriff_fake_attack_1, batch_pred_fake_attack, net, threshold, window_size)

excel_result += f.evaluate_1DCNN(batch_angriff_fake_attack_2, batch_pred_fake_attack, net, threshold, window_size)

######################################################################################################################
excel_result += f.evaluate_1DCNN(batch_angriff_sis_attack_1, batch_pred_sis_attack, net, threshold, window_size)

excel_result += f.evaluate_1DCNN(batch_angriff_sis_attack_2, batch_pred_sis_attack, net, threshold, window_size)

print("End")
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

for i in range(32):
    # Open data from silver bucket
    if i < 16: 
        # open the file in read binary mode
        file = open("C:/Users/lalyor/Documents/Masterarbeit/Signale/Silver/signal_silver_"+str(i), "rb")
        #read the file to numpy array
        arr1 = np.load(file)
        signal_silver += [arr1]
        #close the file
        file.close()
    else: # Open data from black bucket
        # open the file in read binary mode
        file = open("C:/Users/lalyor/Documents/Masterarbeit/Signale/Black/signal_black_"+str(i-16), "rb")
        #read the file to numpy array
        arr1 = np.load(file)
        signal_black += [arr1]
        #close the file
        file.close()

batch_silver = []
batch_black = []

signal_silver = np.array(signal_silver)
signal_black = np.array(signal_black)

for i in range(0,5):
    batch_silver += [signal_silver[:,(i*20):((i+1)*20)]]
    batch_black += [signal_black[:,i*20:(i+1)*20]]
    
######################################################################################################################

dilation_size = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(16,16,3,padding=1)
        self.conv2 = nn.Conv1d(16,16,3,padding=1)
        self.conv3 = nn.Conv1d(16,16,3,padding=1)
        self.conv4 = nn.Conv1d(16,16,3,padding=1)
        self.conv5 = nn.Conv1d(16,16,3,padding=1)
        self.conv6 = nn.Conv1d(48,16,1)
        
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

input_signal_silver = batch_silver[0:4]
prediction_silver = batch_silver[1:5]

input_signal_black = batch_black[0:4]
prediction_black = batch_black[1:5]
    
training_set_silver = Dataset(input_signal_silver, prediction_silver)  #Create the Dataset from the lists
trainloader_silver = torch.utils.data.DataLoader(training_set_silver, batch_size=1,shuffle=False)   #Create random batch of size 16 in order to improve the learning of the network.

training_set_black = Dataset(input_signal_black, prediction_black)  #Create the Dataset from the lists
trainloader_black = torch.utils.data.DataLoader(training_set_black, batch_size=1,shuffle=False)   #Create random batch of size 16 in order to improve the learning of the network.

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
        
        # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
    
    for j, data in enumerate(trainloader_black,0):
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
        
        # print('[%d, %5d] loss: %.3f' % (epoch + 1, j + i + 2, running_loss))
        
    epoch_loss_array += [epoch_loss/8]
    

plt.plot(epoch_loss_array)     
plt.xlabel('Epoch')   
plt.ylabel('Loss')
plt.title('Evolution of the loss through the epoch for 1D-CNN')
plt.show()
   
print("End")
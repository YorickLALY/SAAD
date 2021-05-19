# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:53:47 2021

@author: lalyor
"""


import numpy as np
import matplotlib.pyplot as plt

signal_silver = []
signal_black = []


for i in range(16):
    # Open data from silver bucket
    if i < 8: 
        # open the file in read binary mode
        file = open("C:/Users/lalyor/Documents/Masterarbeit/Runs_8/Silver1/signal_silver_"+str(i), "rb")
        #read the file to numpy array
        arr1 = np.load(file)
        signal_silver += [arr1]
        #close the file
    #     file.close()
    # else: # Open data from black bucket
    #     # open the file in read binary mode
    #     file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Black/signal_black_"+str(i-8), "rb")
    #     #read the file to numpy array
    #     arr1 = np.load(file)
    #     signal_black += [arr1]
    #     #close the file
    #     file.close()
        
t_array = [0.03*i for i in range(len(signal_silver[0]))]        
       

fig, ax = plt.subplots(2,4)
fig.suptitle('Silberner Topf Signale') 

for i in range(8):
    if(i<4):
        ax[0,i].plot(t_array,signal_silver[i])
        ax[0,i].set_ylim([0,3])
    else:
        ax[1,i-4].plot(t_array,signal_silver[i])
        ax[1,i-4].set_ylim([0,3])
    
fig.savefig("C:/Users/lalyor/Documents/Masterarbeit/taster/silver_signal.png",orientation='landscape')
fig.clear()
    
    

# fig, ax = plt.subplots(2,4)
# fig.suptitle('Schwarz Bucket Signal') 

# for i in range(8):
#     if(i<4):
#         ax[0,i].plot(t_array,signal_black[i])
#         ax[0,i].set_ylim([0,3])
#     else:
#         ax[1,i-4].plot(t_array,signal_black[i])
#         ax[1,i-4].set_ylim([0,3])
    
# fig.savefig("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Regroup_rename_attack/black_bucket.png",orientation='landscape')
# fig.clear()
        
        
        

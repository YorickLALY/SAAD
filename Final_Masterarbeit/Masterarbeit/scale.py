# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:51:10 2021

@author: lalyor
"""

import numpy as np
import matplotlib.pyplot as plt

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
        
t_array = [0.1*i for i in range(100)]        
        
for i in range(16):
    fig, ax = plt.subplots()
    ax.set_xlabel('Zeit (s)')
    ax.set_ylabel('Spannungswert (V)')
    fig.suptitle('Signal '+str(i))
    ax.plot(t_array,signal_silver[i])
    plt.ylim([0,3])
    fig.savefig("C:/Users/lalyor/Documents/Masterarbeit/Scale/Silver/signal_silver_"+str(i)+".png",orientation='landscape')
    fig.clear()
    
for i in range(16):
    fig, ax = plt.subplots()
    ax.set_xlabel('Zeit (s)')
    ax.set_ylabel('Spannungswert (V)')
    fig.suptitle('Signal '+str(i))
    ax.plot(t_array,signal_black[i])
    plt.ylim([0,3])
    fig.savefig("C:/Users/lalyor/Documents/Masterarbeit/Scale/Black/signal_black_"+str(i)+".png",orientation='landscape')
    fig.clear()
        
        
        

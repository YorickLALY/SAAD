# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:17:55 2021

@author: lalyor
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import seaborn as sn
import functions as f

######################################################################################################################
# Normal runs extraction

taster = []

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/taster/signal_silver_"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    taster += [arr1]
    #close the file
    file.close()
    

taster = np.array(taster)

######################################################################################################################

signal = []
window_size = 40
threshold = 1

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Run_30_min_8/signal_"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    signal += [arr1]
    #close the file
    file.close()


signal = np.array(signal)

data = np.concatenate((taster,taster,signal,taster), axis = 1)

t_array = [i for i in range(len(data[0]))]

for i in range(8):
    
    #Save signal as an array
    # open a binary file in write mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/data/signal_"+str(i), "wb")
    # save array to the file
    np.save(file, data[i])
    # close the file
    file.close()

    # Plot signal
    fig, ax = plt.subplots()
    
    ax.set_xlabel('Zeit (s)')
    ax.set_ylabel('Spannungswert (V)')
    fig.suptitle('Signal '+str(i))
    ax.plot(t_array,data[i])

    fig.savefig("C:/Users/lalyor/Documents/Masterarbeit/data/signal_"+str(i)+".png",orientation='landscape')
    fig.clear()
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:01:12 2021

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

signal = []

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff_8/Silver/signal_silver_"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    signal += [arr1]
    #close the file
    file.close()


signal = np.array(signal)
    

upscale = [[0.0 for j in range(3*len(signal[0]))] for i in range(len(signal))]
upscale = np.array(upscale)

for i in range(len(signal)):
    for j in range(len(signal[0])):
        for k in range(3):
            upscale[i,3*j+k] = signal[i,j]
            
t_array = [i for i in range(300)]
            
for i in range(8):
    
    #Save signal as an array
    # open a binary file in write mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/fake_angriff_silver/signal_silver_"+str(i), "wb")
    # save array to the file
    np.save(file, upscale[i])
    # close the file
    file.close()

    # Plot signal
    fig, ax = plt.subplots()
    
    ax.set_xlabel('Zeit (s)')
    ax.set_ylabel('Spannungswert (V)')
    fig.suptitle('Signal '+str(i))
    ax.plot(t_array,upscale[i])

    fig.savefig("C:/Users/lalyor/Documents/Masterarbeit/fake_angriff_silver/signal_silver_"+str(i)+".png",orientation='landscape')
    fig.clear()
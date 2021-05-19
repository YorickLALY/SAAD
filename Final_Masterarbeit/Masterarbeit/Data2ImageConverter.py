# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:11:45 2021

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
        
        # Save signal as an image
        fig, ax = plt.subplots()
        
        ax.set_xlabel('Zeit (s)')
        ax.set_ylabel('Spannungswert (V)')
        fig.suptitle('Signal '+str(i))
        ax.plot(t_array,voltage_Signal[i])
        fig.savefig("/media/pi/Volume/Masterarbeit/Black/signal_black_"+str(i)+".png",orientation='landscape')
        fig.clear()
        
    else: # Open data from black bucket
        # open the file in read binary mode
        file = open("C:/Users/lalyor/Documents/Masterarbeit/Signale/Silver/signal_silver_"+str(i-16), "rb")
        #read the file to numpy array
        arr1 = np.load(file)
        signal_black += [arr1]
        #close the file
        file.close()
        
        # Save signal as an image
        fig, ax = plt.subplots()
        
        ax.set_xlabel('Zeit (s)')
        ax.set_ylabel('Spannungswert (V)')
        fig.suptitle('Signal '+str(i))
        ax.plot(t_array,voltage_Signal[i])
        fig.savefig("/media/pi/Volume/Masterarbeit/Black/signal_black_"+str(i)+".png",orientation='landscape')
        fig.clear()
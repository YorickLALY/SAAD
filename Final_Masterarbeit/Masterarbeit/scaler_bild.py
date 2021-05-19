# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:11:25 2021

@author: lalyor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
import functions as f

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

######################################################################################################################

signal = MaxAbsScaler().fit_transform(signal)
# signal = QuantileTransformer(output_distribution='uniform').fit_transform(signal)
# signal = QuantileTransformer(output_distribution='normal').fit_transform(signal)
# signal = PowerTransformer(method='yeo-johnson').fit_transform(signal)
# signal = f.booleanScaler(signal, threshold)

plt.title("BooleanScaler")
plt.xlabel("Signal(i)")
plt.ylabel("Signal(j)")

for i in range(8):
    for j in range(8):  
        plt.scatter(signal[i,:],signal[j,:])
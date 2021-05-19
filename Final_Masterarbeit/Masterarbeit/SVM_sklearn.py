# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:18:22 2021

@author: lalyor
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


######################################################################################################################
# Normal runs extraction

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

signal = signal_silver + signal_black
signal = np.array(signal)

signal_silver = np.array(signal_silver)
signal_black = np.array(signal_black)

batch = np.concatenate((signal_silver[0:8,:],signal_black[0:8,:]),axis=1)

for i in range(1,5):
    batch = np.concatenate((batch,signal_silver[8*i:8*(i+1),:],signal_black[8*i:8*(i+1),:]),axis=1)

batch_silver = []
for i in range(0,980):
    batch_silver += [batch[:,i:(i+20)]]
    
######################################################################################################################
# Attack runs extraction

angriff_silver = []
angriff_black = []

for i in range(16):
    # Open data from silver bucket
    if i < 8:
        # open the file in read binary mode
        file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff/Silver/signal_silver_"+str(i), "rb")
        #read the file to numpy array
        arr1 = np.load(file)
        angriff_silver += [arr1]
        #close the file
        file.close()
    else: # Open data from black bucket
        # open the file in read binary mode
        file = open("C:/Users/lalyor/Documents/Masterarbeit/Angriff/Black/signal_black_"+str(i-8), "rb")
        #read the file to numpy array
        arr1 = np.load(file)
        angriff_black += [arr1]
        #close the file
        file.close()

angriff = angriff_silver + angriff_black
angriff = np.array(angriff)

angriff_silver = np.array(angriff_silver)
angriff_black = np.array(angriff_black)

angriff_batch = np.concatenate((signal_silver[0:8,:],signal_black[0:8,:]),axis=1)

angriff_batch_silver = []

for i in range(0,180):
    angriff_batch_silver += [batch[:,i:(i+20)]]
    
######################################################################################################################
#Without scaling/normalizing

results = [['Scaling Method','Accuracy Test','Accuracy Outliers']]

x_Train = np.concatenate((signal_silver[0:8,:],signal_black[0:8,:]),axis=0)
for i in range(1,5):
    x_Train = np.concatenate((x_Train,signal_silver[8*i:8*(i+1),:],signal_black[8*i:8*(i+1),:]),axis=0)

data = np.concatenate((x_Train,angriff),axis=0)

x_train = x_Train[0:48,:]
x_test = x_Train[48:80,:]
x_outliers = angriff

x_train=np.array(x_train)
x_test=np.array(x_test)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(x_train)


# plt.title("OneClassSVM")

# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

# b1 = plt.scatter(x_train[:, 0], x_train[:, 1], c='white', s=20, edgecolor='k')
# b2 = plt.scatter(x_test[:, 0], x_test[:, 1], c='green',s=20, edgecolor='k')
# c = plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='red', s=20, edgecolor='k')
# plt.axis('tight')
# # plt.xlim((-5, 5))
# # plt.ylim((-5, 5))
# plt.legend([b1, b2, c], ["training observations", "new regular observations", "new abnormal observations"], loc="upper left")
# plt.show()  

y_pred_train_2 = clf.predict(x_train)
y_pred_test_2 = clf.predict(x_test)
y_pred_outliers_2 = clf.predict(x_outliers)

#ONE CLass SVM
print("Without scaling: ")
print("Accuracy test :", list(y_pred_test_2).count(1)/y_pred_test_2.shape[0])
print("Accuracy outliers:", list(y_pred_outliers_2).count(-1)/y_pred_outliers_2.shape[0])

results += [['Without Scaling', list(y_pred_test_2).count(1)/y_pred_test_2.shape[0],list(y_pred_outliers_2).count(-1)/y_pred_outliers_2.shape[0]]] 

######################################################################################################################
#Scaling/ Normalizing

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# StandardScaler
data_train_standard = StandardScaler().fit_transform(x_Train[0:48,:])
data_test_standard = StandardScaler().fit_transform(x_Train[48:80,:])
data_outliers_standard = StandardScaler().fit_transform(angriff)

data_train=np.array(data_train_standard)
data_test=np.array(data_test_standard)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers_standard)

print("StandardScaler: ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['StandardScaler', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# MinMaxScaler
data_train_minmax = MinMaxScaler().fit_transform(x_Train[0:48,:])
data_test_minmax = MinMaxScaler().fit_transform(x_Train[48:80,:])
data_outliers_minmax = MinMaxScaler().fit_transform(angriff)

data_train=np.array(data_train_minmax)
data_test=np.array(data_test_minmax)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers_minmax)

print("MinMaxScaler: ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['MinMaxScaler', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# MaxAbsScaler
data_train = MaxAbsScaler().fit_transform(x_Train[0:48,:])
data_test = MaxAbsScaler().fit_transform(x_Train[48:80,:])
data_outliers = MaxAbsScaler().fit_transform(angriff)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

print("MaxAbsScaler: ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['MaxAbsScaler', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# RobustScaler
data_train = RobustScaler(quantile_range=(25, 75)).fit_transform(x_Train[0:48,:])
data_test = RobustScaler(quantile_range=(25, 75)).fit_transform(x_Train[48:80,:])
data_outliers = RobustScaler(quantile_range=(25, 75)).fit_transform(angriff)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

print("RobustScaler: ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['RobustScaler', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# PowerTransformer method='yeo-johnson'
data_train = PowerTransformer(method='yeo-johnson').fit_transform(x_Train[0:48,:])
data_test = PowerTransformer(method='yeo-johnson').fit_transform(x_Train[48:80,:])
data_outliers = PowerTransformer(method='yeo-johnson').fit_transform(angriff)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

print("PowerTransformer method='yeo-johnson': ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['PowerTransformer method=yeo-johnson', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# # PowerTransformer method='box-cox'
# data_train = PowerTransformer(method='box-cox').fit_transform(x_Train[0:48,:])
# data_test = PowerTransformer(method='box-cox').fit_transform(x_Train[48:80,:])
# data_outliers = PowerTransformer(method='box-cox').fit_transform(angriff)

# data_train=np.array(data_train)
# data_test=np.array(data_test)

# clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
# clf.fit(data_train)

# pred_train = clf.predict(data_train)
# pred_test = clf.predict(data_test)
# pred_outliers = clf.predict(data_outliers)

# print("PowerTransformer method='box-cox': ")
# print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
# print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

# QuantileTransformer output_distribution='uniform'
data_train = QuantileTransformer(output_distribution='uniform').fit_transform(x_Train[0:48,:])
data_test = QuantileTransformer(output_distribution='uniform').fit_transform(x_Train[48:80,:])
data_outliers = QuantileTransformer(output_distribution='uniform').fit_transform(angriff)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

print("QuantileTransformer output_distribution='uniform': ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['QuantileTransformer output_distribution=uniform', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# QuantileTransformer output_distribution='normal'
data_train = QuantileTransformer(output_distribution='normal').fit_transform(x_Train[0:48,:])
data_test = QuantileTransformer(output_distribution='normal').fit_transform(x_Train[48:80,:])
data_outliers = QuantileTransformer(output_distribution='normal').fit_transform(angriff)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

print("QuantileTransformer output_distribution='normal': ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['QuantileTransformer output_distribution=normal', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# Normalizer
data_train = Normalizer().fit_transform(x_Train[0:48,:])
data_test = Normalizer().fit_transform(x_Train[48:80,:])
data_outliers = Normalizer().fit_transform(angriff)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

print("Normalizer: ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['Normalizer', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# Round Value
data_train_dec_2 = np.round(x_Train[0:48,:],2)
data_test_dec_2 = np.round(x_Train[48:80,:],2)
data_outliers_dec_2 = np.round(angriff,2)

data_train=np.array(data_train_dec_2)
data_test=np.array(data_test_dec_2)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train_dec_2)
pred_test = clf.predict(data_test_dec_2)
pred_outliers = clf.predict(data_outliers_dec_2)

print("Round Value to 2 decimals: ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['Round Value', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

# Temporal SVM
data_outliers_dec_2 = np.round(angriff,2)

data_train=np.array(data_train_dec_2)
data_test=np.array(data_test_dec_2)

clf = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf.fit(data_train)

pred_train = clf.predict(data_train_dec_2)
pred_test = clf.predict(data_test_dec_2)
pred_outliers = clf.predict(data_outliers_dec_2)

print("Round Value to 2 decimals: ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['Round Value', list(pred_test).count(1)/pred_test.shape[0],list(pred_outliers).count(-1)/pred_outliers.shape[0]]] 

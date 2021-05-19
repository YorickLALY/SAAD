# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:11:21 2021

@author: lalyor
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix
import functions as f

######################################################################################################################
window_size = 20
signal = []

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/Run_30_min_8/signal_"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    signal += [arr1]
    #close the file
    file.close()


signal = np.array(signal)

# for i in range(0,8):
#     for j in range(0,59400):
#         if (signal[i,j] < 1):
#             signal[i,j] = 0
#         else:
#             signal[i,j] = 1
            
batch = []
for i in range(0,59380):
    batch += [signal[:,i:(i+20)]]

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
    pred_rename_attack_1[i] = -1
    
pred_rename_attack_2 = [1 for i in range(990)] #Replay attack
for i  in range(267,990):
    pred_rename_attack_2[i] = -1
    
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
    pred_replay_attack_1[i] = -1
    
pred_replay_attack_2 = [1 for i in range(990)] #Replay attack
for i  in range(476,990):
    pred_replay_attack_2[i] = -1
    
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

pred_sis_attack = [-1 for i in range(len(angriff_sis_attack_1[0]))] #no change in the behaviour of the process
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
    pred_fake_attack[i] = -1
    
batch_pred_fake_attack = f.create_batch(pred_fake_attack, window_size, pred = 1)
batch_pred_fake_attack = np.array(batch_pred_fake_attack)

######################################################################################################################
#Without scaling/normalizing

# X_lof=np.r_[normal_+2,normal_-2,outliers_]

results = [['Scaling Method','Accuracy Test','Accuracy Outliers']]
result_test_excel = []
result_angriff_excel = []
x_train = signal[:,0:35000]
x_train = np.transpose(x_train)

x_test = signal[:,35000:len(signal[0])]
x_test = np.transpose(x_test)

# x_outliers = np.concatenate((angriff_sis_attack_1,angriff_sis_attack_2), axis = 1)
# x_outliers = np.transpose(x_outliers)
x_outliers = np.transpose(angriff_sis_attack_2)
ground_truth_angriff = pred_sis_attack

neighbours = 3000
print('Neighbours: ', neighbours)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(x_train)

test_pred = lof.predict(x_test)

n_outliers = 0
ground_truth = np.ones(len(x_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()



result_test = (len(x_test)-n_errors)/(len(x_test))
result_test_excel += [result_test]

lof.fit(x_train)
outlier_pred = lof.predict(x_outliers)

n_outliers = len(x_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
result_angriff_excel += [result_outlier]

excel_result = []
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

#LOF
print("No Scaler: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['No Scaler', result_test,result_outlier]] 

# lof_truc = LocalOutlierFactor(n_neighbors=5)
# truc_train = np.transpose(np.concatenate((signal[:,0:35000],angriff_replay_attack_1,angriff_replay_attack_2), axis = 1))
# truc_pred = lof_truc.fit_predict(truc_train)

# n_outliers = len(x_outliers)
# ground_truth = np.ones(len(truc_train), dtype=int)
# ground_truth[-n_outliers:] = -1
# n_errors = (truc_pred != ground_truth).sum()

# cm = confusion_matrix(ground_truth,truc_pred)
# tn, fp, fn, tp = confusion_matrix(ground_truth,truc_pred).ravel()
# r = tn/(tn+fp)
# p = tn/(tn+fn)
# f1 = 2*p*r/(p+r)

# print('Truc = ', (len(truc_train)-n_errors)/(len(truc_train)))

######################################################################################################################
#Scaling/ Normalizing

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# StandardScaler
data_train_standard = StandardScaler().fit_transform(x_train)
data_test_standard = StandardScaler().fit_transform(x_test)
data_outliers_standard = StandardScaler().fit_transform(x_outliers)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train_standard)

test_pred = lof.predict(data_test_standard)

n_outliers = 0
ground_truth = np.ones(len(data_test_standard), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test_standard)-n_errors)/(len(data_test_standard))
result_test_excel += [result_test]


lof.fit(data_train_standard)
outlier_pred = lof.predict(data_outliers_standard)

n_outliers = len(data_outliers_standard)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers_standard) - n_errors)/(len(data_outliers_standard))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("StandardScaler: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['StandardScaler', result_test,result_outlier]] 

# MinMaxScaler
data_train_minmax = MinMaxScaler().fit_transform(x_train)
data_test_minmax = MinMaxScaler().fit_transform(x_test)
data_outliers_minmax = MinMaxScaler().fit_transform(x_outliers)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train_minmax)

test_pred = lof.predict(data_test_minmax)

n_outliers = 0
ground_truth = np.ones(len(data_test_minmax), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test_minmax)-n_errors)/(len(data_test_minmax))
result_test_excel += [result_test]



lof.fit(data_train_minmax)
outlier_pred = lof.predict(data_outliers_minmax)

n_outliers = len(data_outliers_minmax)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers_minmax) - n_errors)/(len(data_outliers_minmax))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("MinMaxScaler: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['MinMaxScaler', result_test,result_outlier]] 

# MaxAbsScaler
data_train = MaxAbsScaler().fit_transform(x_train)
data_test = MaxAbsScaler().fit_transform(x_test)
data_outliers = MaxAbsScaler().fit_transform(x_outliers)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train)

test_pred = lof.predict(data_test)

n_outliers = 0
ground_truth = np.ones(len(data_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test)-n_errors)/(len(data_test))
result_test_excel += [result_test]



lof.fit(data_train)
outlier_pred = lof.predict(data_outliers)

n_outliers = len(data_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers) - n_errors)/(len(data_outliers))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("MaxAbsScaler: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['MaxAbsScaler', result_test,result_outlier]] 

# RobustScaler
data_train = RobustScaler(quantile_range=(25, 75)).fit_transform(x_train)
data_test = RobustScaler(quantile_range=(25, 75)).fit_transform(x_test)
data_outliers = RobustScaler(quantile_range=(25, 75)).fit_transform(x_outliers)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train)

test_pred = lof.predict(data_test)

n_outliers = 0
ground_truth = np.ones(len(data_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test)-n_errors)/(len(data_test))
result_test_excel += [result_test]



lof.fit(data_train)
outlier_pred = lof.predict(data_outliers)

n_outliers = len(data_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers) - n_errors)/(len(data_outliers))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("RobustScaler: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['RobustScaler', result_test,result_outlier]] 

# PowerTransformer method='yeo-johnson'
data_train = PowerTransformer(method='yeo-johnson').fit_transform(x_train)
data_test = PowerTransformer(method='yeo-johnson').fit_transform(x_test)
data_outliers = PowerTransformer(method='yeo-johnson').fit_transform(x_outliers)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train)

test_pred = lof.predict(data_test)

n_outliers = 0
ground_truth = np.ones(len(data_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test)-n_errors)/(len(data_test))
result_test_excel += [result_test]



lof.fit(data_train)
outlier_pred = lof.predict(data_outliers)

n_outliers = len(data_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers) - n_errors)/(len(data_outliers))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("PowerTransformer method='yeo-johnson': ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['PowerTransformer method=yeo-johnson', result_test,result_outlier]] 

# # PowerTransformer method='box-cox'
# data_train = PowerTransformer(method='box-cox').fit_transform(x_train)
# data_test = PowerTransformer(method='box-cox').fit_transform(x_test)
# data_outliers = PowerTransformer(method='box-cox').fit_transform(x_outliers)

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
data_train = QuantileTransformer(output_distribution='uniform').fit_transform(x_train)
data_test = QuantileTransformer(output_distribution='uniform').fit_transform(x_test)
data_outliers = QuantileTransformer(output_distribution='uniform').fit_transform(x_outliers)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train)

test_pred = lof.predict(data_test)

n_outliers = 0
ground_truth = np.ones(len(data_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test)-n_errors)/(len(data_test))
result_test_excel += [result_test]



lof.fit(data_train)
outlier_pred = lof.predict(data_outliers)

n_outliers = len(data_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers) - n_errors)/(len(data_outliers))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("QuantileTransformer output_distribution='uniform': ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['QuantileTransformer output_distribution=uniform', result_test,result_outlier]] 

# QuantileTransformer output_distribution='normal'
data_train = QuantileTransformer(output_distribution='normal').fit_transform(x_train)
data_test = QuantileTransformer(output_distribution='normal').fit_transform(x_test)
data_outliers = QuantileTransformer(output_distribution='normal').fit_transform(x_outliers)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train)

test_pred = lof.predict(data_test)

n_outliers = 0
ground_truth = np.ones(len(data_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test)-n_errors)/(len(data_test))
result_test_excel += [result_test]



lof.fit(data_train)
outlier_pred = lof.predict(data_outliers)

n_outliers = len(data_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers) - n_errors)/(len(data_outliers))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("QuantileTransformer output_distribution='normal': ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['QuantileTransformer output_distribution=normal', result_test,result_outlier]] 

# Normalizer
data_train = Normalizer().fit_transform(x_train)
data_test = Normalizer().fit_transform(x_test)
data_outliers = Normalizer().fit_transform(x_outliers)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train)

test_pred = lof.predict(data_test)

n_outliers = 0
ground_truth = np.ones(len(data_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test)-n_errors)/(len(data_test))
result_test_excel += [result_test]



lof.fit(data_train)
outlier_pred = lof.predict(data_outliers)

n_outliers = len(data_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers) - n_errors)/(len(data_outliers))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("Normalizer: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['Normalizer', result_test,result_outlier]] 

# Round Value
data_train = np.round(x_train,2)
data_test = np.round(x_test,2)
data_outliers = np.round(x_outliers,2)

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train)

test_pred = lof.predict(data_test)

n_outliers = 0
ground_truth = np.ones(len(data_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test)-n_errors)/(len(data_test))
result_test_excel += [result_test]



lof.fit(data_train)
outlier_pred = lof.predict(data_outliers)

n_outliers = len(data_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers) - n_errors)/(len(data_outliers))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("Round Value 2 decimals: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['Round Value 2 decimals', result_test,result_outlier]] 

# Boolean Value
for i in range(0,8):
    for j in range(len(x_train)):
        if (x_train[j,i] < 1):
            data_train[j,i] = 0
        else:
            data_train[j,i] = 1
    for k in range(len(x_test)):
        if (x_test[k,i] < 1):
            data_test[k,i] = 0
        else:
            data_test[k,i] = 1
    for l in range(len(x_outliers)):
        if (x_outliers[l,i] < 1):
            data_outliers[l,i] = 0
        else:
            data_outliers[l,i] = 1

lof = LocalOutlierFactor(n_neighbors=neighbours, novelty = True)
lof.fit(data_train)

test_pred = lof.predict(data_test)

n_outliers = 0
ground_truth = np.ones(len(data_test), dtype=int)
n_errors = (test_pred != ground_truth).sum()

result_test = (len(data_test)-n_errors)/(len(data_test))
result_test_excel += [result_test]



lof.fit(data_train)
outlier_pred = lof.predict(data_outliers)

n_outliers = len(data_outliers)
n_errors = (outlier_pred != ground_truth_angriff).sum()

result_outlier = (len(data_outliers) - n_errors)/(len(data_outliers))
result_angriff_excel += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(outlier_pred[i] == -1):
        if(ground_truth_angriff[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth_angriff[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth_angriff)]
if (true_pos + false_pos) != 0:
    precision += [true_pos / (true_pos + false_pos)]
else:
    precision += [0]
if (true_pos + false_neg) != 0:
    recall += [true_pos / (true_pos + false_neg)] 
else:
    recall += [0]
if (precision[0] + recall[0]) != 0:
    f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
else:
    f1 += [0]

excel_result += accuracy + precision + recall + f1

print("Boolean Value: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

results += [['Boolean Value', result_test,result_outlier]] 

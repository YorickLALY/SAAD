# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 19:23:01 2021

@author: lalyor
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import functions as f


######################################################################################################################
window_size = 40
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

results = [['Scaling Method','Accuracy Test','Accuracy Outliers']]
result_excel_angriff = []
result_excel_test = []
x_train = np.transpose(signal[:,0:35000])
x_test = np.transpose(signal[:,35000:59400])
x_outliers = np.transpose(angriff_sis_attack_1)
ground_truth_angriff = pred_sis_attack

gamma = 0.001
nu = 0.01

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(x_train)


y_pred_train_2 = clf.predict(x_train)
y_pred_test_2 = clf.predict(x_test)
y_pred_outliers_2 = clf.predict(x_outliers)

n_outliers = len(x_outliers)
n_errors = (y_pred_outliers_2 != ground_truth_angriff).sum()
result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
result_excel_angriff += [result_outlier]

excel_result = []
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(y_pred_outliers_2[i] == -1):
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

#ONE CLass SVM
print("Without scaling: ")
print("Accuracy test :", list(y_pred_test_2).count(1)/y_pred_test_2.shape[0])
print("Accuracy outliers:", result_outlier)

results += [['Without Scaling', list(y_pred_test_2).count(1)/y_pred_test_2.shape[0],result_outlier]] 
result_excel_test += [list(y_pred_test_2).count(1)/y_pred_test_2.shape[0]]
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

data_train=np.array(data_train_standard)
data_test=np.array(data_test_standard)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers_standard)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['StandardScaler', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

# MinMaxScaler
data_train_minmax = MinMaxScaler().fit_transform(x_train)
data_test_minmax = MinMaxScaler().fit_transform(x_test)
data_outliers_minmax = MinMaxScaler().fit_transform(x_outliers)

data_train=np.array(data_train_minmax)
data_test=np.array(data_test_minmax)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers_minmax)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['MinMaxScaler', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

# MaxAbsScaler
data_train = MaxAbsScaler().fit_transform(x_train)
data_test = MaxAbsScaler().fit_transform(x_test)
data_outliers = MaxAbsScaler().fit_transform(x_outliers)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['MaxAbsScaler', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

# RobustScaler
data_train = RobustScaler(quantile_range=(25, 75)).fit_transform(x_train)
data_test = RobustScaler(quantile_range=(25, 75)).fit_transform(x_test)
data_outliers = RobustScaler(quantile_range=(25, 75)).fit_transform(x_outliers)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['RobustScaler', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

# PowerTransformer method='yeo-johnson'
data_train = PowerTransformer(method='yeo-johnson').fit_transform(x_train)
data_test = PowerTransformer(method='yeo-johnson').fit_transform(x_test)
data_outliers = PowerTransformer(method='yeo-johnson').fit_transform(x_outliers)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['PowerTransformer method=yeo-johnson', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

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

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['QuantileTransformer output_distribution=uniform', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

# QuantileTransformer output_distribution='normal'
data_train = QuantileTransformer(output_distribution='normal').fit_transform(x_train)
data_test = QuantileTransformer(output_distribution='normal').fit_transform(x_test)
data_outliers = QuantileTransformer(output_distribution='normal').fit_transform(x_outliers)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['QuantileTransformer output_distribution=normal', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

# Normalizer
data_train = Normalizer().fit_transform(x_train)
data_test = Normalizer().fit_transform(x_test)
data_outliers = Normalizer().fit_transform(x_outliers)

data_train=np.array(data_train)
data_test=np.array(data_test)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['Normalizer', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

# Round Value
data_train_dec_2 = np.round(x_train,2)
data_test_dec_2 = np.round(x_test,2)
data_outliers_dec_2 = np.round(x_outliers,2)

data_train=np.array(data_train_dec_2)
data_test=np.array(data_test_dec_2)

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train_dec_2)
pred_test = clf.predict(data_test_dec_2)
pred_outliers = clf.predict(data_outliers_dec_2)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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

print("Round Value to 2 decimals: ")
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['Round Value', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

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

clf = svm.OneClassSVM(kernel='linear', gamma=gamma, nu=nu)
clf.fit(data_train)

pred_train = clf.predict(data_train)
pred_test = clf.predict(data_test)
pred_outliers = clf.predict(data_outliers)

n_outliers = len(pred_outliers)
n_errors = (pred_outliers != ground_truth_angriff).sum()
result_outlier = (len(pred_outliers) - n_errors)/(len(pred_outliers))
result_excel_angriff += [result_outlier]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth_angriff)):
    if(ground_truth_angriff[i] == -1):
        if(pred_outliers[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(pred_outliers[i] == -1):
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
print("Accuracy test :", list(pred_test).count(1)/pred_test.shape[0])
print("Accuracy outliers:", list(pred_outliers).count(-1)/pred_outliers.shape[0])

results += [['Boolean Value', list(pred_test).count(1)/pred_test.shape[0],result_outlier]] 
result_excel_test += [list(pred_test).count(1)/pred_test.shape[0]]

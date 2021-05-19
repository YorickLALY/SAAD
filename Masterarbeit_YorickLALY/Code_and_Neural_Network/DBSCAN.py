# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:09:29 2021

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

for i in range(0,8):
    # open the file in read binary mode
    file = open("C:/Users/lalyor/Documents/Masterarbeit/data/signal_"+str(i), "rb")
    #read the file to numpy array
    arr1 = np.load(file)
    signal += [arr1]
    #close the file
    file.close()


signal = np.array(signal)

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

pred_rename_attack_1 = [0 for i in range(990)] #Replay attack
for i  in range(113,990):
    pred_rename_attack_1[i] = -1
    
pred_rename_attack_2 = [0 for i in range(990)] #Replay attack
for i  in range(267,990):
    pred_rename_attack_2[i] = -1
    

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

pred_replay_attack_1 = [0 for i in range(990)] #Replay attack
for i  in range(395,990):
    pred_replay_attack_1[i] = -1
    
pred_replay_attack_2 = [0 for i in range(990)] #Replay attack
for i  in range(476,990):
    pred_replay_attack_2[i] = -1
    

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

pred_sis_attack = [-1 for i in range(len(angriff_sis_attack_1[0]))] #no change in the behaviour of the process
pred_sis_attack = np.array(pred_sis_attack)

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

pred_fake_attack = [0 for i in range(len(angriff_fake_attack_1[0]))] #Attack when Schranke goes down
pred_fake_attack = np.array(pred_fake_attack)
for i  in range(21*3,46*3):
    pred_fake_attack[i] = -1

######################################################################################################################
# #Without scaling/normalizing

# results = [['Scaling Method','Accuracy Test','Accuracy Outliers']]
# result_excel_angriff = []
# result_excel_test = []

# x_train = signal[:,0:35000]

# x_test = signal
# x_test = np.transpose(x_test)

# x_outliers = np.concatenate((x_train,angriff_fake_attack_1), axis = 1)
# x_outliers = np.transpose(x_outliers)
# # x_outliers = np.transpose(signal[35000:59400])

# dbscan = DBSCAN(eps=1.9, min_samples=3000, algorithm='auto')

# test_pred = dbscan.fit_predict(x_test)

# n_outliers = 0
# ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()

# result_test = (len(x_test)-n_errors)/(len(x_test))
        
# cm_test = confusion_matrix(ground_truth,test_pred)
# cr_test = classification_report(ground_truth,test_pred)


# outlier_pred = dbscan.fit_predict(x_outliers)   

# n_outliers = len(angriff_fake_attack_1[0])
# # ground_truth = np.zeros(35000, dtype=int)
# # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# ground_truth = pred_fake_attack
# n_errors = (ground_truth != outlier_pred[35000:]).sum()

# # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# cr_outlier = classification_report(ground_truth,outlier_pred[35000:])

# true_pos = 0
# true_neg = 0
# false_pos = 0
# false_neg = 0
    
# for i in range(len(ground_truth)):
#     if(outlier_pred[35000 + i] == -1):
#         if(ground_truth[i] == -1):
#             true_pos += 1
#         else:
#             false_pos += 1
#     else:
#         if(ground_truth[i] == -1):
#             false_neg += 1
#         else:
#             true_neg += 1

# accuracy = []
# precision = []
# recall = []
# f1 = []

# accuracy += [(true_pos + true_neg)/len(ground_truth)]
# if (true_pos + false_pos) != 0:
#     precision += [true_pos / (true_pos + false_pos)]
# else:
#     precision += [0]
# if (true_pos + false_neg) != 0:
#     recall += [true_pos / (true_pos + false_neg)] 
# else:
#     recall += [0]
# if (precision[0] + recall[0]) != 0:
#     f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
# else:
#     f1 += [0]

# excel_result = accuracy + precision + recall + f1

# print("No Scaler: ")
# print("Accuracy test :", result_test)
# print("Accuracy outliers:", result_outlier)

# # results += [['No Scaler', result_test,result_outlier]] 
# # result_excel_angriff += [result_outlier]
# # result_excel_test += [result_test]

# # labels = dbscan.labels_

# # # identify core samples
# # core_samples = np.zeros_like(labels, dtype=bool)
# # core_samples[dbscan.core_sample_indices_] = True
# # print(core_samples)
# # # declare the number of clusters
# # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# # print(n_clusters)
# # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # # visualize outputs
# # colors = dbscan.labels_
# # # for i in range(8):
# # #     for j in range(8):  
# # #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)

# ######################################################################################################################
# MinMaxScaler
x_train = signal[:,0:35000]

x_test = signal
x_test = np.transpose(x_test)

x_outliers = np.concatenate((x_train,angriff_replay_attack_1), axis = 1)
x_outliers = np.transpose(x_outliers)
# x_outliers = np.transpose(signal[35000:59400])

dbscan = DBSCAN(eps=0.9, min_samples=750, algorithm='auto')

x_test = MinMaxScaler().fit_transform(x_test)
test_pred = dbscan.fit_predict(x_test)

n_outliers = 0
ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()
n_errors = (test_pred < ground_truth).sum()

result_test = (len(x_test)-n_errors)/(len(x_test))
        
cm_test = confusion_matrix(ground_truth,test_pred)
# cr_test = classification_report(ground_truth,test_pred)


x_outliers = MinMaxScaler().fit_transform(x_outliers)
outlier_pred = dbscan.fit_predict(x_outliers)

n_outliers = len(angriff_fake_attack_1[0])
# ground_truth = np.zeros(35000, dtype=int)
# ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
ground_truth = pred_replay_attack_1
n_errors = (ground_truth != outlier_pred[35000:]).sum()

# result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
cr_outlier = classification_report(ground_truth,outlier_pred[35000:])

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
    
for i in range(len(ground_truth)):
    if(outlier_pred[35000 + i] == -1):
        if(ground_truth[i] == -1):
            true_pos += 1
        else:
            false_pos += 1
    else:
        if(ground_truth[i] == -1):
            false_neg += 1
        else:
            true_neg += 1

accuracy = []
precision = []
recall = []
f1 = []

accuracy += [(true_pos + true_neg)/len(ground_truth)]
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

excel_result = accuracy + precision + recall + f1

print("MinMaxScaler: ")
print("Accuracy test :", result_test)
print("Accuracy outliers:", result_outlier)

# results += [['MinMaxScaler', result_test,result_outlier]] 
# result_excel_angriff += [result_outlier]
# result_excel_test += [result_test]

# labels = dbscan.labels_

# # identify core samples
# core_samples = np.zeros_like(labels, dtype=bool)
# core_samples[dbscan.core_sample_indices_] = True
# print(core_samples)
# # declare the number of clusters
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # visualize outputs
# colors = dbscan.labels_
# # for i in range(8):
# #     for j in range(8):  
# #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)


# ######################################################################################################################
# # StandardScaler
# x_train = signal[:,0:35000]

# x_test = signal
# x_test = np.transpose(x_test)

# x_outliers = np.concatenate((x_train,angriff_fake_attack_1), axis = 1)
# x_outliers = np.transpose(x_outliers)
# # x_outliers = np.transpose(signal[35000:59400])

# dbscan = DBSCAN(eps=3, min_samples=20000, algorithm='auto')

# x_test = StandardScaler().fit_transform(x_test)
# test_pred = dbscan.fit_predict(x_test)

# n_outliers = 0
# ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()

# result_test = (len(x_test)-n_errors)/(len(x_test))
        
# cm_test = confusion_matrix(ground_truth,test_pred)
# # cr_test = classification_report(ground_truth,test_pred)



# x_outliers = StandardScaler().fit_transform(x_outliers)
# outlier_pred = dbscan.fit_predict(x_outliers)

# n_outliers = len(angriff_fake_attack_1[0])
# # ground_truth = np.zeros(35000, dtype=int)
# # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# # ground_truth = np.concatenate((pred_rename_attack_1,pred_rename_attack_2), axis = 0)
# ground_truth = pred_fake_attack
# n_errors = (ground_truth != outlier_pred[35000:]).sum()

# # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# # cr_outlier = classification_report(ground_truth,outlier_pred[35000:])

# true_pos = 0
# true_neg = 0
# false_pos = 0
# false_neg = 0
    
# for i in range(len(ground_truth)):
#     if(outlier_pred[35000 + i] == -1):
#         if(ground_truth[i] == -1):
#             true_pos += 1
#         else:
#             false_pos += 1
#     else:
#         if(ground_truth[i] == -1):
#             false_neg += 1
#         else:
#             true_neg += 1

# accuracy = []
# precision = []
# recall = []
# f1 = []

# accuracy += [(true_pos + true_neg)/len(ground_truth)]
# if (true_pos + false_pos) != 0:
#     precision += [true_pos / (true_pos + false_pos)]
# else:
#     precision += [0]
# if (true_pos + false_neg) != 0:
#     recall += [true_pos / (true_pos + false_neg)] 
# else:
#     recall += [0]
# if (precision[0] + recall[0]) != 0:
#     f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
# else:
#     f1 += [0]

# excel_result = accuracy + precision + recall + f1

# print("StandardScaler: ")
# print("Accuracy test :", result_test)
# print("Accuracy outliers:", result_outlier)

# # results += [['StandardScaler', result_test,result_outlier]] 
# # result_excel_angriff += [result_outlier]
# # result_excel_test += [result_test]

# # labels = dbscan.labels_

# # # identify core samples
# # core_samples = np.zeros_like(labels, dtype=bool)
# # core_samples[dbscan.core_sample_indices_] = True
# # print(core_samples)
# # # declare the number of clusters
# # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# # print(n_clusters)
# # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # # visualize outputs
# # colors = dbscan.labels_
# # # for i in range(8):
# # #     for j in range(8):  
# # #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)

# ######################################################################################################################
# # # BooleanScaler
# # threshold = 1
# # x_train = signal[:,0:35000]

# # x_test = signal
# # x_test = np.transpose(x_test)

# # x_outliers = np.concatenate((x_train,angriff_rename_attack_1,angriff_rename_attack_2), axis = 1)
# # x_outliers = np.transpose(x_outliers)
# # # x_outliers = np.transpose(signal[35000:59400])

# # dbscan = DBSCAN(eps=0.90, min_samples=12000, algorithm='auto')

# # x_test = f.booleanScaler(x_test, threshold)
# # test_pred = dbscan.fit_predict(x_test)

# # n_outliers = 0
# # ground_truth = np.zeros(len(x_test), dtype=int)
# # n_errors = (test_pred != ground_truth).sum()

# # result_test = (len(x_test)-n_errors)/(len(x_test))
        
# # cm_test = confusion_matrix(ground_truth,test_pred)
# # cr_test = classification_report(ground_truth,test_pred)





# # x_outliers = f.booleanScaler(x_outliers, threshold)
# # outlier_pred = dbscan.fit_predict(x_outliers)

# # n_outliers = 2*len(angriff_rename_attack_1[0])
# # # ground_truth = np.zeros(35000, dtype=int)
# # # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# # ground_truth = np.concatenate((pred_rename_attack_1,pred_rename_attack_2), axis = 0)
# # n_errors = (ground_truth != outlier_pred[35000:]).sum()

# # # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# # result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# # cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# # cr_outlier = classification_report(ground_truth,outlier_pred[35000:])


# # print("BooleanScaler: ")
# # print("Accuracy test :", result_test)
# # print("Accuracy outliers:", result_outlier)

# # results += [['BooleanScaler', result_test,result_outlier]] 
# result_excel_angriff += [result_outlier]
# result_excel_test += [result_test]

# # labels = dbscan.labels_

# # # identify core samples
# # core_samples = np.zeros_like(labels, dtype=bool)
# # core_samples[dbscan.core_sample_indices_] = True
# # print(core_samples)
# # # declare the number of clusters
# # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# # print(n_clusters)
# # # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # # visualize outputs
# # colors = dbscan.labels_
# # # for i in range(8):
# # #     for j in range(8):  
# # #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)

# ######################################################################################################################
# # RobustScaler
# x_train = signal[:,0:35000]

# x_test = signal
# x_test = np.transpose(x_test)

# x_outliers = np.concatenate((x_train,angriff_replay_attack_2), axis = 1)
# x_outliers = np.transpose(x_outliers)
# # x_outliers = np.transpose(signal[35000:59400])

# dbscan = DBSCAN(eps=3, min_samples=20000, algorithm='auto')

# x_test = RobustScaler().fit_transform(x_test)
# test_pred = dbscan.fit_predict(x_test)

# n_outliers = 0
# ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()

# result_test = (len(x_test)-n_errors)/(len(x_test))
        
# cm_test = confusion_matrix(ground_truth,test_pred)
# cr_test = classification_report(ground_truth,test_pred)





# x_outliers = RobustScaler().fit_transform(x_outliers)
# outlier_pred = dbscan.fit_predict(x_outliers)

# n_outliers = len(angriff_rename_attack_1[0])
# # ground_truth = np.zeros(35000, dtype=int)
# # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# # ground_truth = np.concatenate((pred_rename_attack_1,pred_rename_attack_2), axis = 0)
# ground_truth = pred_replay_attack_2
# n_errors = (ground_truth != outlier_pred[35000:]).sum()

# true_pos = 0
# true_neg = 0
# false_pos = 0
# false_neg = 0
    
# for i in range(len(ground_truth)):
#     if(outlier_pred[35000 + i] == -1):
#         if(ground_truth[i] == -1):
#             true_pos += 1
#         else:
#             false_pos += 1
#     else:
#         if(ground_truth[i] == -1):
#             false_neg += 1
#         else:
#             true_neg += 1

# accuracy = []
# precision = []
# recall = []
# f1 = []

# accuracy += [(true_pos + true_neg)/len(ground_truth)]
# if (true_pos + false_pos) != 0:
#     precision += [true_pos / (true_pos + false_pos)]
# else:
#     precision += [0]
# if (true_pos + false_neg) != 0:
#     recall += [true_pos / (true_pos + false_neg)] 
# else:
#     recall += [0]
# if (precision[0] + recall[0]) != 0:
#     f1 += [2*precision[0]*recall[0] / (precision[0] + recall[0])]  
# else:
#     f1 += [0]

# excel_result = accuracy + precision + recall + f1

# # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# cr_outlier = classification_report(ground_truth,outlier_pred[35000:])


# print("RobustScaler: ")
# print("Accuracy test :", result_test)
# print("Accuracy outliers:", result_outlier)

# # results += [['RobustScaler', result_test,result_outlier]] 

# # labels = dbscan.labels_

# # # identify core samples
# # core_samples = np.zeros_like(labels, dtype=bool)
# # core_samples[dbscan.core_sample_indices_] = True
# # print(core_samples)
# # # declare the number of clusters
# # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# # print(n_clusters)
# # print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # # visualize outputs
# # colors = dbscan.labels_
# # # for i in range(8):
# # #     for j in range(8):  
# # #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)

# ######################################################################################################################
# # MaxAbsScaler
# x_train = signal[:,0:35000]

# x_test = signal
# x_test = np.transpose(x_test)

# x_outliers = np.concatenate((x_train,angriff_rename_attack_1,angriff_rename_attack_2), axis = 1)
# x_outliers = np.transpose(x_outliers)
# # x_outliers = np.transpose(signal[35000:59400])

# dbscan = DBSCAN(eps=0.80, min_samples=2000, algorithm='auto')

# x_test = MaxAbsScaler().fit_transform(x_test)
# test_pred = dbscan.fit_predict(x_test)

# n_outliers = 0
# ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()

# result_test = (len(x_test)-n_errors)/(len(x_test))
        
# cm_test = confusion_matrix(ground_truth,test_pred)
# cr_test = classification_report(ground_truth,test_pred)





# x_outliers = MaxAbsScaler().fit_transform(x_outliers)
# outlier_pred = dbscan.fit_predict(x_outliers)

# n_outliers = 2*len(angriff_rename_attack_1[0])
# # ground_truth = np.zeros(35000, dtype=int)
# # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# ground_truth = np.concatenate((pred_rename_attack_1,pred_rename_attack_2), axis = 0)
# n_errors = (ground_truth != outlier_pred[35000:]).sum()

# # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# cr_outlier = classification_report(ground_truth,outlier_pred[35000:])


# print("MaxAbsScaler: ")
# print("Accuracy test :", result_test)
# print("Accuracy outliers:", result_outlier)

# results += [['MaxAbsScaler', result_test,result_outlier]] 

# labels = dbscan.labels_

# # identify core samples
# core_samples = np.zeros_like(labels, dtype=bool)
# core_samples[dbscan.core_sample_indices_] = True
# print(core_samples)
# # declare the number of clusters
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # visualize outputs
# colors = dbscan.labels_
# # for i in range(8):
# #     for j in range(8):  
# #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)

# ######################################################################################################################
# # Normalizer
# x_train = signal[:,0:35000]

# x_test = signal
# x_test = np.transpose(x_test)

# x_outliers = np.concatenate((x_train,angriff_rename_attack_1,angriff_rename_attack_2), axis = 1)
# x_outliers = np.transpose(x_outliers)
# # x_outliers = np.transpose(signal[35000:59400])

# dbscan = DBSCAN(eps=0.80, min_samples=2000, algorithm='auto')

# x_test = Normalizer().fit_transform(x_test)
# test_pred = dbscan.fit_predict(x_test)

# n_outliers = 0
# ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()

# result_test = (len(x_test)-n_errors)/(len(x_test))
        
# cm_test = confusion_matrix(ground_truth,test_pred)
# cr_test = classification_report(ground_truth,test_pred)





# x_outliers = Normalizer().fit_transform(x_outliers)
# outlier_pred = dbscan.fit_predict(x_outliers)

# n_outliers = 2*len(angriff_rename_attack_1[0])
# # ground_truth = np.zeros(35000, dtype=int)
# # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# ground_truth = np.concatenate((pred_rename_attack_1,pred_rename_attack_2), axis = 0)
# n_errors = (ground_truth != outlier_pred[35000:]).sum()

# # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# cr_outlier = classification_report(ground_truth,outlier_pred[35000:])


# print("Normalizer: ")
# print("Accuracy test :", result_test)
# print("Accuracy outliers:", result_outlier)

# results += [['Normalizer', result_test,result_outlier]] 

# labels = dbscan.labels_

# # identify core samples
# core_samples = np.zeros_like(labels, dtype=bool)
# core_samples[dbscan.core_sample_indices_] = True
# print(core_samples)
# # declare the number of clusters
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # visualize outputs
# colors = dbscan.labels_
# # for i in range(8):
# #     for j in range(8):  
# #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)

# ######################################################################################################################
# # QuantileTransformer(output_distribution='uniform')
# x_train = signal[:,0:35000]

# x_test = signal
# x_test = np.transpose(x_test)

# x_outliers = np.concatenate((x_train,angriff_rename_attack_1,angriff_rename_attack_2), axis = 1)
# x_outliers = np.transpose(x_outliers)
# # x_outliers = np.transpose(signal[35000:59400])

# dbscan = DBSCAN(eps=0.90, min_samples=2000, algorithm='auto')

# x_test = QuantileTransformer(output_distribution='uniform').fit_transform(x_test)
# test_pred = dbscan.fit_predict(x_test)

# n_outliers = 0
# ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()

# result_test = (len(x_test)-n_errors)/(len(x_test))
        
# cm_test = confusion_matrix(ground_truth,test_pred)
# cr_test = classification_report(ground_truth,test_pred)





# x_outliers = QuantileTransformer(output_distribution='uniform').fit_transform(x_outliers)
# outlier_pred = dbscan.fit_predict(x_outliers)

# n_outliers = 2*len(angriff_rename_attack_1[0])
# # ground_truth = np.zeros(35000, dtype=int)
# # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# ground_truth = np.concatenate((pred_rename_attack_1,pred_rename_attack_2), axis = 0)
# n_errors = (ground_truth != outlier_pred[35000:]).sum()

# # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# cr_outlier = classification_report(ground_truth,outlier_pred[35000:])


# print("QuantileTransformer(output_distribution='uniform'): ")
# print("Accuracy test :", result_test)
# print("Accuracy outliers:", result_outlier)

# results += [['QuantileTransformer(output_distribution=uniform)', result_test,result_outlier]] 

# labels = dbscan.labels_

# # identify core samples
# core_samples = np.zeros_like(labels, dtype=bool)
# core_samples[dbscan.core_sample_indices_] = True
# print(core_samples)
# # declare the number of clusters
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # visualize outputs
# colors = dbscan.labels_
# # for i in range(8):
# #     for j in range(8):  
# #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)

# ######################################################################################################################
# # QuantileTransformer(output_distribution='normal')
# x_train = signal[:,0:35000]

# x_test = signal
# x_test = np.transpose(x_test)

# x_outliers = np.concatenate((x_train,angriff_rename_attack_1,angriff_rename_attack_2), axis = 1)
# x_outliers = np.transpose(x_outliers)
# # x_outliers = np.transpose(signal[35000:59400])

# dbscan = DBSCAN(eps=1.50, min_samples=2000, algorithm='auto')

# x_test = QuantileTransformer(output_distribution='normal').fit_transform(x_test)
# test_pred = dbscan.fit_predict(x_test)

# n_outliers = 0
# ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()

# result_test = (len(x_test)-n_errors)/(len(x_test))
        
# cm_test = confusion_matrix(ground_truth,test_pred)
# cr_test = classification_report(ground_truth,test_pred)





# x_outliers = QuantileTransformer(output_distribution='normal').fit_transform(x_outliers)
# outlier_pred = dbscan.fit_predict(x_outliers)

# n_outliers = 2*len(angriff_rename_attack_1[0])
# # ground_truth = np.zeros(35000, dtype=int)
# # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# ground_truth = np.concatenate((pred_rename_attack_1,pred_rename_attack_2), axis = 0)
# n_errors = (ground_truth != outlier_pred[35000:]).sum()

# # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# cr_outlier = classification_report(ground_truth,outlier_pred[35000:])


# print("QuantileTransformer(output_distribution='normal'): ")
# print("Accuracy test :", result_test)
# print("Accuracy outliers:", result_outlier)

# results += [['QuantileTransformer(output_distribution=normal)', result_test,result_outlier]] 

# labels = dbscan.labels_

# # identify core samples
# core_samples = np.zeros_like(labels, dtype=bool)
# core_samples[dbscan.core_sample_indices_] = True
# print(core_samples)
# # declare the number of clusters
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # visualize outputs
# colors = dbscan.labels_
# # for i in range(8):
# #     for j in range(8):  
# #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)

# ######################################################################################################################
# # PowerTransformer(method='yeo-johnson')
# x_train = signal[:,0:35000]

# x_test = signal
# x_test = np.transpose(x_test)

# x_outliers = np.concatenate((x_train,angriff_rename_attack_1,angriff_rename_attack_2), axis = 1)
# x_outliers = np.transpose(x_outliers)
# # x_outliers = np.transpose(signal[35000:59400])

# dbscan = DBSCAN(eps=1.50, min_samples=2000, algorithm='auto')

# x_test = PowerTransformer(method='yeo-johnson').fit_transform(x_test)
# test_pred = dbscan.fit_predict(x_test)

# n_outliers = 0
# ground_truth = np.zeros(len(x_test), dtype=int)
# n_errors = (test_pred != ground_truth).sum()

# result_test = (len(x_test)-n_errors)/(len(x_test))
        
# cm_test = confusion_matrix(ground_truth,test_pred)
# cr_test = classification_report(ground_truth,test_pred)





# x_outliers = PowerTransformer(method='yeo-johnson').fit_transform(x_outliers)
# outlier_pred = dbscan.fit_predict(x_outliers)

# n_outliers = 2*len(angriff_rename_attack_1[0])
# # ground_truth = np.zeros(35000, dtype=int)
# # ground_truth = np.concatenate((ground_truth,pred_sis_attack,pred_sis_attack), axis = 0)
# ground_truth = np.concatenate((pred_rename_attack_1,pred_rename_attack_2), axis = 0)
# n_errors = (ground_truth != outlier_pred[35000:]).sum()

# # result_outlier = (len(x_outliers) - n_errors)/(len(x_outliers))
# result_outlier = (len(ground_truth) - n_errors)/(len(ground_truth))

# cm_outlier = confusion_matrix(ground_truth,outlier_pred[35000:])
# cr_outlier = classification_report(ground_truth,outlier_pred[35000:])


# print("PowerTransformer(method='yeo-johnson'): ")
# print("Accuracy test :", result_test)
# print("Accuracy outliers:", result_outlier)

# results += [['PowerTransformer(method=yeo-johnson)', result_test,result_outlier]] 

# labels = dbscan.labels_

# # identify core samples
# core_samples = np.zeros_like(labels, dtype=bool)
# core_samples[dbscan.core_sample_indices_] = True
# print(core_samples)
# # declare the number of clusters
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(n_clusters)
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_outliers, labels))

# # visualize outputs
# colors = dbscan.labels_
# # for i in range(8):
# #     for j in range(8):  
# #         plt.scatter(x_outliers[:,i],x_outliers[:,j], c = colors)
# ######################################################################################################################

# # labels_true = ground_truth
# # X = x_outliers
# # core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
# # core_samples_mask[dbscan.core_sample_indices_] = True
# # # Number of clusters in labels, ignoring noise if present.
# # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# # n_noise_ = list(labels).count(-1)

# # print('Estimated number of clusters: %d' % n_clusters_)
# # print('Estimated number of noise points: %d' % n_noise_)
# # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# # print("Adjusted Rand Index: %0.3f"
# #       % metrics.adjusted_rand_score(labels_true, labels))
# # print("Adjusted Mutual Information: %0.3f"
# #       % metrics.adjusted_mutual_info_score(labels_true, labels))
# # print("Silhouette Coefficient: %0.3f"
# #       % metrics.silhouette_score(X, labels))

# # # #############################################################################
# # # Plot result

# # # Black removed and is used for noise instead.
# # unique_labels = set(labels)
# # colors = [plt.cm.Spectral(each)
# #           for each in np.linspace(0, 1, len(unique_labels))]
# # for k, col in zip(unique_labels, colors):
# #     if k == -1:
# #         # Black used for noise.
# #         col = [0, 0, 0, 1]

# #     class_member_mask = (labels == k)

# #     xy = X[class_member_mask & core_samples_mask]
# #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
# #              markeredgecolor='k', markersize=14)

# #     xy = X[class_member_mask & ~core_samples_mask]
# #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
# #              markeredgecolor='k', markersize=6)

# # plt.title('Estimated number of clusters: %d' % n_clusters_)
# # plt.show()
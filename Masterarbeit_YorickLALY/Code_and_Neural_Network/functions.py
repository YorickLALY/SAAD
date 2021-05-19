# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:17:46 2021

@author: lalyor
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def create_batch(signal, window_size, pred = 0):
    batch = []
    if pred == 0:
        for i in range(len(signal[0])-window_size):
            batch += [signal[:,i:(i+window_size)]]
    else:
        for i in range(len(signal)-window_size):
            batch += [signal[:][i:(i+window_size)]]
    return batch

def booleanScaler(signal, threshold):
    s = [[0 for i in range(len(signal[0]))] for j in range(len(signal))]
    s = np.array(s)
    for i in range(len(signal)):
        for j in range(len(signal[0])):
            if (signal[i,j] < threshold):
                s[i,j] = 0
            else:
                s[i,j] = 1
    return s

def evaluate_autoencoder(batch, batch_pred, net, threshold, window_size, classifier=False, threshold_c=0):
    
    true_pos = [0 for i in range(8)]
    true_neg = [0 for i in range(8)]
    false_pos = [0 for i in range(8)]
    false_neg = [0 for i in range(8)]
    
    for i in range(len(batch)-window_size):
        input_signal = torch.Tensor([batch[i]])
        pred = input_signal
        input_signal = input_signal.flatten()
        pred = pred.flatten()
        
        output = net(input_signal)
        
        output = output.detach().numpy()
        pred = pred.detach().numpy()
        compare = abs(output - pred)
        
        if classifier == False:
            for j in range(len(compare)):
                if (compare[j] >= threshold):
                    if batch_pred[i,j//window_size] == 0:
                        true_pos[j//window_size] += 1
                    else:
                        false_pos[j//window_size] += 1
                else:
                    if batch_pred[i,j//window_size] == 1:
                        true_neg[j//window_size] += 1
                    else:
                        false_neg[j//window_size] += 1
        else:
            sensors = [0.0 for k in range(8)]
            sensors = np.array(sensors)
            
            for k in range(len(sensors)):
                sensors[k] = sqrt(mean_squared_error(output[(k*window_size):(k+1)*window_size], pred[(k*window_size):(k+1)*window_size])) 
                
            anomaly_window = [0 for k in range(8)]
            
            for j in range(len(compare)):
                if batch_pred[i,j//window_size] == 0:
                    anomaly_window[j//window_size] += 1
                    
            for j in range(len(anomaly_window)):
                if sensors[j] >= threshold_c:
                    if anomaly_window[j] != 0:
                        true_pos[j] += 40
                    else:
                        false_pos[j] += 40
                else:
                    if anomaly_window[j] != 0:
                        false_neg[j] += 40
                    else:
                        true_neg[j] += 40
            
            # anomaly_point = [0 for k in range(8)]
            # anomaly_window = [0 for k in range(8)]
            
            # for j in range(len(compare)):
            #     if batch_pred[i,j//window_size] == 0:
            #         anomaly_window[j//window_size] += 1
            #     if (compare[j] >= threshold):
            #         anomaly_point[j//window_size] += 1
            # for j in range(len(anomaly_point)):
            #     if anomaly_point[j] >= threshold_c:
            #         if anomaly_window[j] != 0:
            #             true_pos[j] += 40
            #         else:
            #             false_pos[j] += 40
            #     else:
            #         if anomaly_window[j] != 0:
            #             false_neg[j] += 40
            #         else:
            #             true_neg[j] += 40
    
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for i in range(8):
        accuracy += [(true_pos[i] + true_neg[i])/((len(batch)-window_size)*window_size)]
        if (true_pos[i] + false_pos[i]) != 0:
            precision += [true_pos[i] / (true_pos[i] + false_pos[i])]
        else:
            precision += [0]
        if (true_pos[i] + false_neg[i]) != 0:
            recall += [true_pos[i] / (true_pos[i] + false_neg[i])] 
        else:
            recall += [0]
        if (precision[i] + recall[i]) != 0:
            f1 += [2*precision[i]*recall[i] / (precision[i] + recall[i])]  
        else:
            f1 += [0]
            
    sum_a = 0
    sum_p = 0
    sum_r = 0
    sum_f = 0
    
    for i in range(8):
        sum_a += accuracy[i]
        sum_p += precision[i]
        sum_r += recall[i]
        sum_f += f1[i]
        
    accuracy += [sum_a/8]
    precision += [sum_p/8]
    recall += [sum_r/8]
    f1 += [sum_f/8]
    
    excel_result = accuracy + precision + recall + f1
    return excel_result
    
def evaluate_1DCNN(batch, batch_pred, net, threshold, window_size):
    true_pos = [0 for i in range(8)]
    true_neg = [0 for i in range(8)]
    false_pos = [0 for i in range(8)]
    false_neg = [0 for i in range(8)]
    
    for i in range(len(batch)-window_size):
        input_signal = torch.Tensor([batch[i]])
        
        output = net(input_signal)
        
        output = output.detach().numpy()
        output = output[0]
        compare = abs(output - batch[i+window_size]) 
        
        for j in range(len(compare)):
            for k in range(len(compare[0])):
                if (compare[j,k] > threshold):
                    if batch_pred[i+window_size,k] == 0:
                        true_pos[j] += 1
                    else:
                        false_pos[j] += 1
                else:
                    if batch_pred[i+window_size,k] == 1:
                        true_neg[j] += 1
                    else:
                        false_neg[j] += 1
    
    accuracy = []
    precision = []
    recall = []
    f1 = []
    
    for i in range(8):
        accuracy += [(true_pos[i] + true_neg[i])/((len(batch)-window_size)*window_size)]
        if (true_pos[i] + false_pos[i]) != 0:
            precision += [true_pos[i] / (true_pos[i] + false_pos[i])]
        else:
            precision += [0]
        if (true_pos[i] + false_neg[i]) != 0:
            recall += [true_pos[i] / (true_pos[i] + false_neg[i])] 
        else:
            recall += [0]
        if (precision[i] + recall[i]) != 0:
            f1 += [2*precision[i]*recall[i] / (precision[i] + recall[i])]  
        else:
            f1 += [0]
    
    sum_a = 0
    sum_p = 0
    sum_r = 0
    sum_f = 0
    
    for i in range(8):
        sum_a += accuracy[i]
        sum_p += precision[i]
        sum_r += recall[i]
        sum_f += f1[i]
        
    accuracy += [sum_a/8]
    precision += [sum_p/8]
    recall += [sum_r/8]
    f1 += [sum_f/8]
    
    excel_result = accuracy + precision + recall + f1
    return excel_result
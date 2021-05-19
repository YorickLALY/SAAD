# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:53:09 2021

@author: lalyor
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import seaborn as sn
import functions as f
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

######################################################################################################################
# Normal runs extraction

signal = []

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
signal_desc = []
for i in range(8):   
    signal_desc += [pd.Series(signal[i,:])]
    
corr_silver = pd.DataFrame(signal[0:8,:])
corr_silver = corr_silver.transpose()

desc_corr_silver = corr_silver.describe()

corr_mat = corr_silver.corr()

cov_mat = corr_silver.cov()

######################################################################################################################
signal_bool = f.booleanScaler(signal,1)
signal_desc_bool = []

for i in range(8):   
    signal_desc_bool += [pd.Series(signal_bool[i,:])]
    
corr_silver_bool = pd.DataFrame(signal_bool)
corr_silver_bool = corr_silver_bool.transpose()

desc_corr_silver_bool = corr_silver_bool.describe()

corr_mat_bool = corr_silver_bool.corr()

cov_mat_bool = corr_silver_bool.cov()

######################################################################################################################
signal_StandardScaler = StandardScaler().fit_transform(signal)
signal_desc_StandardScaler = []

for i in range(8):   
    signal_desc_StandardScaler += [pd.Series(signal_StandardScaler[i,:])]
    
corr_silver_StandardScaler = pd.DataFrame(signal_StandardScaler)
corr_silver_StandardScaler = corr_silver_StandardScaler.transpose()

desc_corr_silver_StandardScaler = corr_silver_StandardScaler.describe()

corr_mat_StandardScaler = corr_silver_StandardScaler.corr()

cov_mat_StandardScaler = corr_silver_StandardScaler.cov()
    
######################################################################################################################
signal_MinMaxScaler = MinMaxScaler().fit_transform(signal)
signal_desc_MinMaxScaler = []

for i in range(8):   
    signal_desc_MinMaxScaler += [pd.Series(signal_MinMaxScaler[i,:])]
    
corr_silver_MinMaxScaler = pd.DataFrame(signal_MinMaxScaler)
corr_silver_MinMaxScaler = corr_silver_MinMaxScaler.transpose()

desc_corr_silver_MinMaxScaler = corr_silver_MinMaxScaler.describe()

corr_mat_MinMaxScaler= corr_silver_MinMaxScaler.corr()

cov_mat_MinMaxScaler = corr_silver_MinMaxScaler.cov()

# comp_signal_MinMaxScaler = []
# for i in range(8):
#     reshape_signal = (signal[i]).reshape(1,-1)
#     comp_signal_MinMaxScaler += [MinMaxScaler().fit_transform(reshape_signal)]

# comp_signal_MinMaxScaler = np.array(comp_signal_MinMaxScaler)
# comp_signal_MinMaxScaler= np.transpose(comp_signal_MinMaxScaler,(1, 0, 2))
# comp_signal_MinMaxScaler = comp_signal_MinMaxScaler[0]

# comp_signal_desc_MinMaxScaler = []
# for i in range(8):   
#     comp_signal_desc_MinMaxScaler += [pd.Series(comp_signal_MinMaxScaler[i,:])]
    
# comp_corr_silver_MinMaxScaler = pd.DataFrame(comp_signal_desc_MinMaxScaler)
# comp_corr_silver_MinMaxScaler = comp_corr_silver_MinMaxScaler.transpose()

# comp_desc_corr_silver_MinMaxScaler = comp_corr_silver_MinMaxScaler.describe()

# comp_corr_mat_MinMaxScaler = comp_corr_silver_MinMaxScaler.corr()

# comp_cov_mat_MinMaxScaler = comp_corr_silver_MinMaxScaler.cov()

######################################################################################################################
signal_MaxAbsScaler = MaxAbsScaler().fit_transform(signal)
signal_desc_MaxAbsScaler = []

for i in range(8):   
    signal_desc_MaxAbsScaler += [pd.Series(signal_MaxAbsScaler[i,:])]
    
corr_silver_MaxAbsScaler = pd.DataFrame(signal_MaxAbsScaler)
corr_silver_MaxAbsScaler = corr_silver_MaxAbsScaler.transpose()

desc_corr_silver_MaxAbsScaler = corr_silver_MaxAbsScaler.describe()

corr_mat_MaxAbsScaler = corr_silver_MaxAbsScaler.corr()

cov_mat_MaxAbsScaler = corr_silver_MaxAbsScaler.cov()
    
######################################################################################################################
signal_RobustScaler = RobustScaler().fit_transform(signal)
signal_desc_RobustScaler = []

for i in range(8):   
    signal_desc_RobustScaler += [pd.Series(signal_RobustScaler[i,:])]
    
corr_silver_RobustScaler = pd.DataFrame(signal_RobustScaler)
corr_silver_RobustScaler = corr_silver_RobustScaler.transpose()

desc_corr_silver_RobustScaler = corr_silver_RobustScaler.describe()

corr_mat_RobustScaler = corr_silver_RobustScaler.corr()

cov_mat_RobustScaler = corr_silver_RobustScaler.cov()
    
######################################################################################################################
signal_Normalizer = Normalizer().fit_transform(signal)
signal_desc_Normalizer = []

for i in range(8):   
    signal_desc_Normalizer += [pd.Series(signal_Normalizer[i,:])]
    
corr_silver_Normalizer = pd.DataFrame(signal_Normalizer)
corr_silver_Normalizer = corr_silver_Normalizer.transpose()

desc_corr_silver_Normalizer = corr_silver_Normalizer.describe()

corr_mat_Normalizer = corr_silver_Normalizer.corr()

cov_mat_Normalizer = corr_silver_Normalizer.cov()
    
######################################################################################################################
signal_PowerTransformer = PowerTransformer(method='yeo-johnson').fit_transform(signal)
signal_desc_PowerTransformer = []

for i in range(8):   
    signal_desc_PowerTransformer += [pd.Series(signal_PowerTransformer[i,:])]
    
corr_silver_PowerTransformer = pd.DataFrame(signal_PowerTransformer)
corr_silver_PowerTransformer = corr_silver_PowerTransformer.transpose()

desc_corr_silver_PowerTransformer = corr_silver_PowerTransformer.describe()

corr_mat_PowerTransformer = corr_silver_PowerTransformer.corr()

cov_mat_PowerTransformer = corr_silver_PowerTransformer.cov()
    
######################################################################################################################
signal_QuantileTransformerUniform = QuantileTransformer(output_distribution='uniform').fit_transform(signal)
signal_desc_QuantileTransformerUniform = []

for i in range(8):   
    signal_desc_QuantileTransformerUniform += [pd.Series(signal_QuantileTransformerUniform[i,:])]
    
corr_silver_QuantileTransformerUniform = pd.DataFrame(signal_QuantileTransformerUniform)
corr_silver_QuantileTransformerUniform = corr_silver_QuantileTransformerUniform.transpose()

desc_corr_silver_QuantileTransformerUniform = corr_silver_QuantileTransformerUniform.describe()

corr_mat_QuantileTransformerUniform = corr_silver_QuantileTransformerUniform.corr()

cov_mat_QuantileTransformerUniform = corr_silver_QuantileTransformerUniform.cov()
    
######################################################################################################################
signal_QuantileTransformerNormal = QuantileTransformer(output_distribution='normal').fit_transform(signal)
signal_desc_QuantileTransformerNormal = []

for i in range(8):   
    signal_desc_QuantileTransformerNormal += [pd.Series(signal_QuantileTransformerNormal[i,:])]
    
corr_silver_QuantileTransformerNormal = pd.DataFrame(signal_QuantileTransformerNormal)
corr_silver_QuantileTransformerNormal = corr_silver_QuantileTransformerNormal.transpose()

desc_corr_silver_QuantileTransformerNormal = corr_silver_QuantileTransformerNormal.describe()

corr_mat_QuantileTransformerNormal = corr_silver_QuantileTransformerNormal.corr()

cov_mat_QuantileTransformerNormal = corr_silver_QuantileTransformerNormal.cov()
    
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
    
angriff_rename_attack_desc_1 = []
for i in range(8):   
    angriff_rename_attack_desc_1 += [(pd.Series(angriff_rename_attack_1[i,:])).describe()]
    
angriff_rename_attack_desc_2 = []
for i in range(8):   
    angriff_rename_attack_desc_2 += [(pd.Series(angriff_rename_attack_2[i,:])).describe()]
    
corr_angriff_rename_attack_1 = pd.DataFrame(angriff_rename_attack_1)
corr_angriff_rename_attack_1 = corr_angriff_rename_attack_1.transpose()

desc_corr_angriff_rename_attack_1 = corr_angriff_rename_attack_1.describe()

corr_mat_angriff_rename_attack_1 = corr_angriff_rename_attack_1.corr()

cov_mat_angriff_rename_attack_1 = corr_angriff_rename_attack_1.cov()

corr_angriff_rename_attack_2 = pd.DataFrame(angriff_rename_attack_2)
corr_angriff_rename_attack_2 = corr_angriff_rename_attack_2.transpose()

desc_corr_angriff_rename_attack_2 = corr_angriff_rename_attack_2.describe()

corr_mat_angriff_rename_attack_2 = corr_angriff_rename_attack_2.corr()

cov_mat_angriff_rename_attack_2 = corr_angriff_rename_attack_2.cov()

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

angriff_replay_attack_desc_1 = []
for i in range(8):   
    angriff_replay_attack_desc_1 += [(pd.Series(angriff_replay_attack_1[i,:])).describe()]
    
angriff_replay_attack_desc_2 = []
for i in range(8):   
    angriff_replay_attack_desc_2 += [(pd.Series(angriff_replay_attack_2[i,:])).describe()]
    
corr_angriff_replay_attack_1 = pd.DataFrame(angriff_replay_attack_1)
corr_angriff_replay_attack_1 = corr_angriff_replay_attack_1.transpose()

desc_corr_angriff_replay_attack_1 = corr_angriff_replay_attack_1.describe()

corr_mat_angriff_replay_attack_1 = corr_angriff_replay_attack_1.corr()

cov_mat_angriff_replay_attack_1 = corr_angriff_replay_attack_1.cov()

corr_angriff_replay_attack_2 = pd.DataFrame(angriff_replay_attack_2)
corr_angriff_replay_attack_2 = corr_angriff_replay_attack_2.transpose()

desc_corr_angriff_replay_attack_2 = corr_angriff_replay_attack_2.describe()

corr_mat_angriff_replay_attack_2 = corr_angriff_replay_attack_2.corr()

cov_mat_angriff_replay_attack_2 = corr_angriff_replay_attack_2.cov()

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

angriff_sis_attack_desc_1 = []
for i in range(8):   
    angriff_sis_attack_desc_1 += [(pd.Series(angriff_sis_attack_1[i,:])).describe()]
    
angriff_sis_attack_desc_2 = []
for i in range(8):   
    angriff_sis_attack_desc_2 += [(pd.Series(angriff_sis_attack_2[i,:])).describe()]
    
corr_angriff_sis_attack_1 = pd.DataFrame(angriff_sis_attack_1)
corr_angriff_sis_attack_1 = corr_angriff_sis_attack_1.transpose()

desc_corr_angriff_sis_attack_1 = corr_angriff_sis_attack_1.describe()

corr_mat_angriff_sis_attack_1 = corr_angriff_sis_attack_1.corr()

cov_mat_angriff_sis_attack_1 = corr_angriff_sis_attack_1.cov()

corr_angriff_sis_attack_2 = pd.DataFrame(angriff_sis_attack_2)
corr_angriff_sis_attack_2 = corr_angriff_sis_attack_2.transpose()

desc_corr_angriff_sis_attack_2 = corr_angriff_sis_attack_2.describe()

corr_mat_angriff_sis_attack_2 = corr_angriff_sis_attack_2.corr()

cov_mat_angriff_sis_attack_2 = corr_angriff_sis_attack_2.cov()

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

angriff_fake_attack_desc_1 = []
for i in range(8):   
    angriff_fake_attack_desc_1 += [(pd.Series(angriff_fake_attack_1[i,:])).describe()]
    
angriff_fake_attack_desc_2 = []
for i in range(8):   
    angriff_fake_attack_desc_2 += [(pd.Series(angriff_fake_attack_2[i,:])).describe()]
    
corr_angriff_fake_attack_1 = pd.DataFrame(angriff_fake_attack_1)
corr_angriff_fake_attack_1 = corr_angriff_fake_attack_1.transpose()

desc_corr_angriff_fake_attack_1 = corr_angriff_fake_attack_1.describe()

corr_mat_angriff_fake_attack_1 = corr_angriff_fake_attack_1.corr()

cov_mat_angriff_fake_attack_1 = corr_angriff_fake_attack_1.cov()

corr_angriff_fake_attack_2 = pd.DataFrame(angriff_fake_attack_2)
corr_angriff_fake_attack_2 = corr_angriff_fake_attack_2.transpose()

desc_corr_angriff_fake_attack_2 = corr_angriff_fake_attack_2.describe()

corr_mat_angriff_fake_attack_2 = corr_angriff_fake_attack_2.corr()

cov_mat_angriff_fake_attack_2 = corr_angriff_fake_attack_2.cov()

######################################################################################################################

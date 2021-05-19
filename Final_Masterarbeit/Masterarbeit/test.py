# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 14:28:15 2021

@author: lalyor
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor

######################################################################################################################

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


signal_silver = np.array(signal_silver)
signal_black = np.array(signal_black)

batch = np.concatenate((signal_silver[0:8,:],signal_black[0:8,:]),axis=1)

for i in range(1,5):
    batch = np.concatenate((batch,signal_silver[8*i:8*(i+1),:],signal_black[8*i:8*(i+1),:]),axis=1)

batch_silver = []
for i in range(0,980):
    batch_silver += [batch[:,i:(i+20)]]


######################################################################################################################

outliers = np.array(signal_black)
normal = batch

x_lof = np.r_[normal,outliers]


from sklearn.decomposition import PCA

pca=PCA(n_components=2)
x_train_pca=pca.fit_transform(x_lof)
x_test_pca=pca.fit_transform(normal)
x_outliers_pca=pca.fit_transform(outliers)

x_train_pca=pd.DataFrame(x_train_pca)
x_test_pca=pd.DataFrame(x_test_pca)
x_outliers_pca=pd.DataFrame(x_outliers_pca)

plt.scatter(x_train_pca[0],x_train_pca[1],label="Normal Train")
plt.scatter(x_outliers_pca[0],x_outliers_pca[1],label='Outlier')
plt.scatter(x_test_pca[0],x_test_pca[1],label="Normal Test")
plt.legend()
plt.show()

X_train = normal
X_test = normal
X_outliers = outliers
clf2=svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf2.fit(X_train)
# plot the line, the samples, and the nearest vectors to the plane
x_train=X_train[:10000,[1,28]]
x_test=X_test[:1000,[1,28]]
x_outliers=X_outliers[:100,[1,28]]

x_train=np.array(x_train)
x_test=np.array(x_test)
x_outliers=np.array(x_outliers)

clf_test_2 = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)
clf_test_2.fit(x_train)

xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf_test_2.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("OneClassSVM")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(x_train[:, 0], x_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(x_test[:, 0], x_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
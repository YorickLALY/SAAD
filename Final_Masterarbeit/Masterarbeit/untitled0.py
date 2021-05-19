# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:51:28 2021

@author: lalyor
"""

from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np


output = [0 for i in range(200)]
pred = [1 for j in range(200)]
output = np.array(output)
pred = np.array(pred)
sensors = sqrt(mean_squared_error(output[0:40], pred[0:40])) 


sensors[k] = sqrt(mean_squared_error(output[(k*window_size):(k+1)*window_size], pred[(k*window_size):(k+1)*window_size]))
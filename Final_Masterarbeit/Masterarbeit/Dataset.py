# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:19:42 2021

@author: lalyor
"""

import torch

from torch.utils import data

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, input_signal, prediction):
        'Initialization'
        self.prediction = prediction
        self.input_signal = input_signal
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_signal)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID_i = self.input_signal[index]
        ID_p = self.prediction[index]
        
        # Load data
        I = torch.Tensor(ID_i)
        P = torch.Tensor(ID_p)
            
        return I,P
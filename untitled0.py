# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:20:01 2024

@author: PC
"""

import pickle

# Function to load data from a pickle file
def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the data from the uploaded file
filename = r'C:\mscode\test\seoul_challenge\data\2024-05-02_15-19-26.pkl'
loaded_data = load_from_pickle(filename)
print(loaded_data)

import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.array(loaded_data)[:,3])

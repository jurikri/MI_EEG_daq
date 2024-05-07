# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:37:59 2024

@author: PC
"""

import os
import sys
# sys.path.append(r'E:\EEGsaves\stroop_0423_WIS 박람회')
import glob
import pickle
# import ranking_board_backend
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
    

X, Y = [], []
SR = 250
tlen = int(SR*10)

#%%

names = ['t1', 't2', 't3']

for n in names:
    directory_path = r'\mscode\test\seoul_challenge\data'
    
    pkl_files = glob.glob(os.path.join(directory_path + '\\' + n, '*.pkl'))
    for j in tqdm(range(len(pkl_files))):
        with open(pkl_files[j], 'rb') as file:
            msdict = pickle.load(file)
            
        if j == 0: eeg_df = np.array(msdict)
        if j > 0: eeg_df = np.concatenate((eeg_df, np.array(msdict)), axis=0)
    
    msbins = np.arange(0, eeg_df.shape[0]-tlen, 250, dtype=int)
    
    for j in range(len(msbins)):
        xtmp2 = eeg_df[msbins[j]:msbins[j]+tlen,:][:,[0,1]]
        if xtmp2.shape[0] == tlen:
            xtmp3 = []
            for ch in range(2):
                xtmp3.append(xtmp2[:,ch]/np.mean(xtmp2[:, ch]))
    
            X.append(xtmp3)
            ytmp = [None, None]
            if n in ['t1', 't2']: ytmp = [1,0]
            if n in ['t3']: ytmp = [0,1]
            Y.append(ytmp)
            
#%%
            
X = np.array(X); Y = np.array(Y)   
print(X.shape)
print(np.mean(Y, axis=0))
#%%
c0 = np.where(Y[:,0]==1)[0]
c1 = np.where(Y[:,1]==1)[0]

np.mean(X[c0])
np.mean(X[c1])

#%%

import numpy as np
from scipy.fft import fft, fftfreq

SR = 250  # Sampling rate
DATA_LENGTH = int(SR*10)  # Length of the EEG data

beta1, beta2 = 1, 1

lsave = []
for beta1 in tqdm(np.arange(-2, 2, 0.1)):
    for beta2 in np.arange(-2, 2, 0.1):
        mssave = []
        for i in range(len(X)):
            eeg_data = (X[i,0,:] * beta1) + (X[i,1,:] * beta2)
            eeg_fft = fft(eeg_data)
            frequencies = fftfreq(DATA_LENGTH, 1/DATA_LENGTH)
            magnitude_spectrum = np.abs(eeg_fft)
            power_spectrum = magnitude_spectrum ** 2
            
            mask = frequencies > 0  # Remove negative frequencies if needed
            mask &= frequencies <= 50  # Limit to 50 Hz
            
            filtered_frequencies = frequencies[mask]
            filtered_power_spectrum = power_spectrum[mask]
            mssave.append(filtered_power_spectrum /np.mean(filtered_power_spectrum ))
        
        
        mssave = np.array(mssave)
        
        roc_save = []
        for i in range(5,15):
            _, roc_auc, _,_ = msFunction.msROC(mssave[c0][:,i], mssave[c1][:,i], repeat=2)
            roc_save.append(roc_auc)
        
        loss_value = -np.max(roc_save)
        lsave.append([beta1, beta2, loss_value])
        print(loss_value)

#%%
lsave = np.array(lsave)
minix = np.argmin(lsave[:,2])
beta1, beta2 = lsave[minix, 0], lsave[minix, 1]

# beta1, beta2 = 1, 1

mssave = []
for i in range(len(X)):
    eeg_data = (X[i,0,:] * beta1) + (X[i,1,:] * beta2)
    eeg_fft = fft(eeg_data)
    frequencies = fftfreq(DATA_LENGTH, 1/SR)
    magnitude_spectrum = np.abs(eeg_fft)
    power_spectrum = magnitude_spectrum ** 2
    
    mask = frequencies > 0  # Remove negative frequencies if needed
    mask &= frequencies <= 50  # Limit to 50 Hz
    
    filtered_frequencies = frequencies[mask]
    filtered_power_spectrum = power_spectrum[mask]
    mssave.append(filtered_power_spectrum /np.mean(filtered_power_spectrum ))

mssave = np.array(mssave)

plt.plot(np.mean(mssave[c0], axis=0)[:30])
plt.plot(np.mean(mssave[c1], axis=0)[:30])

#%%

from scipy.optimize import minimize, Bounds
import numpy as np
import msFunction  # Ensure msFunction is defined or imported correctly

# Constants
SR = 250  # Sampling rate
DATA_LENGTH = 250  # Length of the EEG data

# Function to compute the loss value
def compute_loss(params, X, c0, c1):
    beta1, beta2 = params
    mssave = []
    for i in range(len(X)):
        eeg_data = (X[i, :, 0] * beta1) + (X[i, :, 0] * beta2)
        eeg_fft = fft(eeg_data)
        frequencies = fftfreq(DATA_LENGTH, 1 / SR)
        magnitude_spectrum = np.abs(eeg_fft)
        power_spectrum = magnitude_spectrum ** 2
        
        mask = (frequencies > 0) & (frequencies <= 50)
        
        filtered_power_spectrum = power_spectrum[mask]
        mssave.append(filtered_power_spectrum / np.mean(filtered_power_spectrum))

    mssave = np.array(mssave)
    roc_save = []
    for i in range(5, 15):
        _, roc_auc, _, _ = msFunction.msROC(mssave[c0][:, i], mssave[c1][:, i], repeat=2)
        roc_save.append(roc_auc)

    loss_value = -np.max(roc_save)
    return loss_value

# Optimization function with bounds and options for more iterations
def optimize_betas(X, c0, c1, beta_bounds, max_iter):
    result = minimize(
        compute_loss,
        [1, 1],  # Initial guess
        args=(X, c0, c1),
        method='L-BFGS-B',
        bounds=beta_bounds,
        options={'maxiter': max_iter, 'disp': True}
    )
    return result.x, result.fun

# Specify bounds for beta1 and beta2
beta_bounds = Bounds([-10, -10], [10, 10])  # Here, 0 <= beta1 <= 2 and 0 <= beta2 <= 2

# Call the optimization function
optimal_betas, optimal_loss = optimize_betas(X, c0, c1, beta_bounds, max_iter=10000)
print("Optimal betas:", optimal_betas)
print("Optimal loss value:", optimal_loss)

























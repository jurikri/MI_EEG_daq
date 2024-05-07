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
import sys; 
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction
    


directory_path = r'C:\mscode\test\seoul_challenge\data'
pkl_files = glob.glob(os.path.join(directory_path, '*.pkl'))
for j in tqdm(range(len(pkl_files))):
    with open(pkl_files[j], 'rb') as file:
        msdict = pickle.load(file)
        
    if j == 0: eeg_df = np.array(msdict)
    if j > 0: eeg_df = np.concatenate((eeg_df, np.array(msdict)), axis=0)

print(eeg_df.shape)

directory_path = r'C:\mscode\test\seoul_challenge\data_instruction'
pkl_files = glob.glob(os.path.join(directory_path, '*.pkl'))
for j in tqdm(range(len(pkl_files))):
    with open(pkl_files[j], 'rb') as file:
        msdict = pickle.load(file)
        
    if j == 0: instruction_df = np.array(msdict)
    if j > 0: instruction_df = np.concatenate((instruction_df, np.array(msdict)), axis=0)

print(instruction_df.shape)

#%%
SR = 250
tix = eeg_df[:,-1]
template = np.arange(0, 5, 1/100)
X, Y = [], []
for i in tqdm(range(len(instruction_df))):
    label = instruction_df[i][1]
    tp = instruction_df[i][2]
    
    s = float(tp)
    e = float(tp) + 5

    vix = np.where(np.logical_and((tix > s), (tix < e)))[0]
    if len(vix) > 0:
        vt = tix[vix]
        vix2 = vix[np.argsort(vt)]
        
        xtmp = eeg_df[vix2][:,[0,1]]
        xtmp3 = []
        for ch in range(2):
            xtmp3.append(xtmp[:,ch]/np.mean(xtmp[:, ch]))

        

        if label == 'Release':
            ytmp = [1, 0]
        elif label == 'Grasp':
            ytmp = [0, 1]
            
        xtmp3 = np.array(xtmp3)[:,:1200]
        X.append(xtmp3)
        Y.append(ytmp)
        
X = np.array(X); Y = np.array(Y)   
print(X.shape)

#%%

c0 = np.where(Y[:,0]==1)[0]
c1 = np.where(Y[:,1]==1)[0]

import numpy as np
from scipy.fft import fft, fftfreq

SR = 250  # Sampling rate
DATA_LENGTH = 1200  # Length of the EEG data

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
        for i in range(8,20):
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


#%%
import numpy as np
import matplotlib.pyplot as plt

# 평균과 SEM 계산
mean_c0 = np.mean(mssave[c0], axis=0)[:30]
sem_c0 = np.std(mssave[c0], axis=0, ddof=1)[:30] / np.sqrt(len(mssave[c0]))

mean_c1 = np.mean(mssave[c1], axis=0)[:30]
sem_c1 = np.std(mssave[c1], axis=0, ddof=1)[:30] / np.sqrt(len(mssave[c1]))

# x축 값 생성
x_values = np.arange(30)

# 평균 그리기
plt.plot(x_values, mean_c0, label='Group C0', color='blue')
plt.plot(x_values, mean_c1, label='Group C1', color='red')

# SEM 색상 영역으로 표현
plt.fill_between(x_values, mean_c0 - sem_c0, mean_c0 + sem_c0, color='blue', alpha=0.3)
plt.fill_between(x_values, mean_c1 - sem_c1, mean_c1 + sem_c1, color='red', alpha=0.3)

# 그래프 세팅
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Mean and SEM of Groups C0 and C1')
plt.legend()
plt.show()
























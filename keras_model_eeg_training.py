# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:00:27 2024

@author: PC
"""


import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

import sys; 
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction
import matlab_FIRfilter
import ms_EEG_wavelet
from tqdm import tqdm
import math
import time
import os
import pickle
# import ray
import random
from scipy.signal import find_peaks


#%%
SR = 250
import sys
sys.path.append(r'C:\mscode\test\seoul_challenge')
import keras_model_eeg
kera_model = keras_model_eeg.msmain(time_points=X[0].shape[0], channels=X[0].shape[1], sampling_rate=SR)

#%%

hist = kera_model.fit(X, Y, batch_size=2**5, epochs=100, shuffle=True)

#%%
# kera_model.summary()
#%%
from tensorflow.keras.models import Model
import sys; 
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction

# 특정 입력 A 정의

c1 = np.where(Y[:,0]==1)[0]
c2 = np.where(Y[:,1]==1)[0]

c1_input = X[c1]  # 예: 테스트 데이터
c2_input = X[c2]
# kera_model에서 concatenated_fi_output 레이어의 출력을 추출하기 위한 새 모델 생성
# 이 모델은 kera_model의 가중치와 구조를 공유합니다.
intermediate_layer_model = Model(inputs=kera_model.input, 
                                  outputs=kera_model.get_layer('concatenated_fi_output').output)  # 'concatenate_x'는 해당 레이어의 이름으로 교체해야 합니다.

# 특정 입력 A에 대한 중간 레이어의 출력값 계산
intermediate_output = intermediate_layer_model.predict(X)
# intermediate_output_c2 = intermediate_layer_model.predict(c2_input)
# 결과 출력
# print(intermediate_output.shape)
fi_plot_c1 = np.mean(intermediate_output[c1], axis=0)
fi_plot_c2 = np.mean(intermediate_output[c2], axis=0)
plt.plot(fi_plot_c1)
plt.plot(fi_plot_c2)

mssave = []
for i in range(49):
    _, roc_auc, _,_ = msFunction.msROC(intermediate_output[c1][i], intermediate_output[c2][i])
    mssave.append(roc_auc)

plt.figure()
plt.plot(mssave)
#%%

Yhat = kera_model.predict(X)
print(np.mean(Yhat[c1][:,1]), np.mean(Yhat[c2][:,1]))

acc,roc, _ ,_ = msFunction.msROC(Yhat[c1][:,1], Yhat[c2][:,1])
print(acc,roc)
    













































# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:07:44 2024

@author: sistri
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime

# DOWNLOAD AND PREPROCESS THE DATA



data_PIOT_1 = pd.read_csv('./Data/prova1DEMO2023.csv')
t_start_long = data_PIOT_1['timestamp'].iloc[1500]
t_end_long = data_PIOT_1['timestamp'].iloc[-450]
data_PIOT_1_long = data_PIOT_1[(data_PIOT_1['timestamp'] >= t_start_long) & (data_PIOT_1['timestamp'] <= t_end_long)]
time_PIOT_long = data_PIOT_1_long['timestamp']
time_PIOT_long= pd.to_datetime(time_PIOT_long, unit='ms')
delay = time_PIOT_long.diff()
ACP_1 = -data_PIOT_1_long['REG_LINE_ACP_1'] - data_PIOT_1_long['REG_LINE_ACP_2'] - data_PIOT_1_long['REG_LINE_ACP_3']
freq_1 = data_PIOT_1_long['REG_FREQUENCY']

# Second experiment

data_PIOT_2 = pd.read_csv('./Data/prova1_exp_20231201.csv')
t_start_long = data_PIOT_2['timestamp'].iloc[-1100]
t_end_long = data_PIOT_2['timestamp'].iloc[-500]
data_PIOT_2_long = data_PIOT_2[(data_PIOT_2['timestamp'] >= t_start_long) & (data_PIOT_2['timestamp'] <= t_end_long)]
time_PIOT_long = data_PIOT_2_long['timestamp']
time_PIOT_long= pd.to_datetime(time_PIOT_long, unit='ms')
ACP_2 = -data_PIOT_2_long['REG_LINE_ACP_1'] - data_PIOT_2_long['REG_LINE_ACP_2'] - data_PIOT_2_long['REG_LINE_ACP_3']
freq_2 = data_PIOT_2_long['REG_FREQUENCY']

max(freq_2)
max(ACP_2)
# NORMALIZATION AND ANALYSIS OF THE ERROR
# Normalization for the first test

freq_1_clamped = np.clip(freq_1, 49.9, 50.1)
min_freq, max_freq = 49.9, 50.1
min_ACP, max_ACP = 12.25, 17.75
normalized_freq_1 = ((freq_1_clamped - min_freq) / (max_freq - min_freq)) * (max_ACP - min_ACP) + min_ACP

shift_amount_1 = 0  # Number of positions to shift
normalized_freq_shifted_1 = np.roll(normalized_freq_1, shift_amount_1)
min_length = min(len(normalized_freq_shifted_1), len(ACP_1))
normalized_freq_shifted_1 = normalized_freq_shifted_1[:min_length]
ACP_1_adjusted = ACP_1[:min_length]


error_1= np.array(ACP_1_adjusted) - np.array(normalized_freq_shifted_1)
mean_error_1 = np.mean(error_1)
mean_mean_error_1 = error_1/len(ACP_1_adjusted)
rmse_1 = np.sqrt(mean_error_1**2)
average_error_1 = rmse_1 / len(ACP_1)
#print('error'error_1)
print('np.mean:', mean_error_1)
#print('np.meanmean:' , mean_mean_error_1)

plt.figure(figsize=(10, 6))
plt.plot(range(0,len(normalized_freq_1[:min_length])),normalized_freq_1[:min_length])
plt.plot(range(0,len(ACP_1_adjusted[:min_length])),ACP_1_adjusted[:min_length])


# Normalization for the second test
freq_2_clamped = np.clip(freq_2, 49.9, 50.1)
min_freq, max_freq = 49.9, 50.1
min_ACP, max_ACP = 12.25, 16.75
normalized_freq_2 = ((freq_2_clamped - min_freq) / (max_freq - min_freq)) * (max_ACP - min_ACP) + min_ACP

shift_amount_2 = 0  # Number of positions to shift
normalized_freq_shifted_2 = np.roll(normalized_freq_2, shift_amount_2)
min_length = min(len(normalized_freq_shifted_2), len(ACP_2))
normalized_freq_shifted_2 = normalized_freq_shifted_2[:min_length]
ACP_2_adjusted = ACP_2[:min_length]


error_2= np.array(ACP_2_adjusted) - np.array(normalized_freq_shifted_2)
mean_error_2 = np.mean(error_2)
mean_mean_error_2 = error_2/len(ACP_2_adjusted)
rmse_2 = np.sqrt(mean_error_2**2)
average_error_2 = rmse_2 / len(ACP_2)

plt.figure(figsize=(10, 6))
plt.plot(range(0,len(normalized_freq_2[:min_length])),normalized_freq_2[:min_length])
plt.plot(range(0,len(ACP_2_adjusted[:min_length])),ACP_2_adjusted[:min_length])

print('np.mean:', mean_error_2)
#print('np.meanmean:' , mean_mean_error_2)




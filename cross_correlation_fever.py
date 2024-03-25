# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:34:05 2024

@author: sistri
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pandas as pd
import scipy.stats as stats
# Set layout constants

DefFontSize = 20
EVSEcolors = ['blue', 'red', 'green']
t_start_test = pd.to_datetime('2024-01-16 10:33:50')
t_end_test = pd.to_datetime('2024-01-16 11:15:05')
P_FL = 19
P_bid=3
k = 2*P_bid/0.2

# Define the file path (you'll need to set this to your file's path)
prova1 = pd.read_csv('./Data/test_2024_01_16/prova1_20240116.csv')
prova1['timestamp']= pd.to_datetime(prova1['timestamp'], unit='ms')
prova1 = prova1[(prova1['timestamp'] >= t_start_test) & (prova1['timestamp'] <= t_end_test)]
prova1['P_f_ref'] = (P_FL - P_bid) + k*(prova1['REG_FREQUENCY']- 50)

ACP = -prova1['REG_LINE_ACP_1'] - prova1['REG_LINE_ACP_2'] - prova1['REG_LINE_ACP_3']
time = prova1['timestamp']
freq = prova1['P_f_ref']
ACP_centered = ACP - np.mean(ACP)
freq_centered = freq - np.mean(freq)
time_delta = (time.diff().dt.total_seconds().mean())




# Compute the cross-correlation
#cross_correlation = sm.tsa.stattools.ccf(ACP_centered, freq_centered, adjusted=False)
cross_correlation = np.correlate(ACP_centered, freq_centered, mode='full')
corr_coeff = np.corrcoef(ACP_centered, freq_centered)
normalization_factor = np.std(ACP) * np.std(freq) * len(ACP)
normalized_cross_correlation = cross_correlation / normalization_factor
lags = np.arange(-(len(freq) - 1), len(freq))
time_intervals = lags * time_delta
time_intervals2 = lags * time_delta
plt.figure()
plt.plot(time_intervals, normalized_cross_correlation, color = 'blue')
plt.xlabel('Lag [s]')
plt.ylabel('Cross-correlation')
plt.tight_layout()
plt.grid(True)
#plt.savefig('./graphs/Cross_corr_normalized.pdf')

lag_at_max_corr = lags[np.argmax(normalized_cross_correlation)]
lag_at_max_corr_time = time_intervals[np.argmax(normalized_cross_correlation)]
print("The lag is:", lag_at_max_corr, "time units")
print()
print("Corrcoeff is :", corr_coeff)

# Shift ACP by 7 positions towards the beginning
shift = 8
shifted_ACP = np.roll(ACP, - shift)

# Compute the error between the shifted ACP and freq
error = shifted_ACP - freq
print(np.mean(error))
# Plotting the error distribution
plt.figure(figsize=(10, 6))
plt.hist(error, bins=50, density=True, alpha=0.6, color='g')
plt.title('Error Distribution between Shifted ACP and Freq')
plt.xlabel('Error')
plt.ylabel('Density')
plt.show()
plt.savefig('./graphs/error_graph.pdf')

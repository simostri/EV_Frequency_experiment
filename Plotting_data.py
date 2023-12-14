# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:49:28 2023

@author: sistri
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime

# Load the data
frequency_df = pd.read_csv('./Data/Frequency-data-2023-09-21.csv',skiprows=2,names=['Time', 'frequency'])
power_df = pd.read_csv('./Data/Active Power-data-2023-09-21 21 20 28.csv',skiprows=2,names=['Time', 'power'])

# Preprocessing the data
frequency_df['frequency'] = frequency_df['frequency'].str.replace(';$', '', regex=True)
frequency_df['frequency'] = frequency_df['frequency'].astype(float)

#frequency_df['Time'] = pd.to_datetime(frequency_df['Time'], unit='s')
fig, ax1 = plt.subplots()

# Plot frequency and power on the same graph with 2 y axes
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Frequency', color=color)
ax1.plot(frequency_df['Time'], frequency_df['frequency'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Power', color=color)
ax2.plot(power_df['Time'], -power_df['power'], color=color)
ax2.set_ylim(4.5,5.75)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout() 
plt.show()

#frequency_df['Time']= int(frequency_df['Time']/1000)
# Convert epoch time in milliseconds to datetime
#frequency_df['Time'] = pd.to_datetime(frequency_df['Time'], unit='ms')
#power_df['Time'] = pd.to_datetime(power_df['Time'], unit='ms')
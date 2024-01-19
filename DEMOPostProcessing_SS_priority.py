# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:51:18 2023

@author: klape
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime

# Set layout constants
plt.close('all')
plt.rcParams['font.size'] = 20 
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
#plt.rcdefaults()
#plt.rcParams['grid.color'] = 'white'
#plt.rcParams['grid.linestyle'] = '-'
#plt.rcParams['grid.linewidth'] = 0.5
#plt.rcParams['grid.alpha'] = 1.0

DefFontSize = 20

# DTU colors
EVSEcolors = ['blue', 'red', 'green']

# Retrieve data
data_PIOT = pd.read_csv('./Data/test_2024_01_16/prova1_20240116.csv')
data_CH = pd.read_csv('./Data/test_2024_01_16/Log_1705405987_CH_acdc-352656103199330.csv')

t_start_CH = data_CH['timestamp']
t_end_CH = data_CH['timestamp']
data_CH = data_CH[(data_CH['timestamp'] >= t_start_CH) & (data_CH['timestamp'] <= t_end_CH)]
time_CH = data_CH['timestamp']
time_CH= pd.to_datetime(time_CH, unit='ms')

t_start_long = data_PIOT['timestamp'].iloc[-450]
t_end_long = data_PIOT['timestamp'].iloc[-1]
data_PIOT_long = data_PIOT[(data_PIOT['timestamp'] >= t_start_long) & (data_PIOT['timestamp'] <= t_end_long)]
time_PIOT_long = data_PIOT_long['timestamp']
time_PIOT_long= pd.to_datetime(time_PIOT_long, unit='ms')

t_start_short = data_PIOT['timestamp'].iloc[-450]
t_end_short = data_PIOT['timestamp'].iloc[-1]
data_PIOT_short = data_PIOT[(data_PIOT['timestamp'] >= t_start_short) & (data_PIOT['timestamp'] <= t_end_short)]
time_PIOT_short = data_PIOT_short['timestamp']
time_PIOT_short= pd.to_datetime(time_PIOT_short, unit='ms')


plug_state = np.zeros((6, len(data_PIOT['Plug1_status'])))



ACP = -data_PIOT_long['REG_LINE_ACP_1'] - data_PIOT_long['REG_LINE_ACP_2'] - data_PIOT_long['REG_LINE_ACP_3']
freq = data_PIOT_long['REG_FREQUENCY']

cross_correlation = np.correlate(ACP - np.mean(ACP),freq- np.mean(freq),  mode='full')
lags = np.arange(-(len(freq) - 1), len(freq))

plt.figure(figsize=(10, 6))
plt.plot(lags, cross_correlation - 1.58)
plt.xlim(0, 30) 

plt.title('Cross-correlation between Curve 1 and Curve 2')
plt.xlabel('Lags')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.savefig('./graphs/Cross_corr_priority.pdf')

rmse = np.sqrt(np.mean((np.array(ACP) - np.array(freq))**2))
average_error = rmse / len(ACP)


# Create a figure with three subplots
fig, (ax1, ax3, ax4) = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True)

# First subplot
ax1.grid(True)
line1, = ax1.plot(time_PIOT_long, data_PIOT_long['REG_FREQUENCY'], label='Measured frequency', color=EVSEcolors[0])
ax1.set_ylim(49.9, 50.1) 
ax1.set_ylabel('Frequency [Hz]', fontsize=DefFontSize * 1, color=EVSEcolors[0])
ax2 = ax1.twinx()
line2, = ax2.plot(time_PIOT_long, -data_PIOT_long['REG_LINE_ACP_1'] - data_PIOT_long['REG_LINE_ACP_2'] - data_PIOT_long['REG_LINE_ACP_3'], color=EVSEcolors[1], label='Cluster Power')
ax2.set_ylim(13, 19)
ax2.set_ylabel('Cluster power [kW]', fontsize=DefFontSize * 1, color=EVSEcolors[1])
ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="best")

# Second subplot
ax3.grid(True)
ax3.plot(time_PIOT_long, data_PIOT_long['Plug1_priority_broadcast'], label='Priority of EV1', color=EVSEcolors[0])
ax3.plot(time_PIOT_long, data_PIOT_long['Plug2_priority_broadcast'], label='Priority of EV2', color=EVSEcolors[1])
ax3.set_ylabel('Priority of the EV', fontsize=DefFontSize * 1)
ax3.legend(loc="best")

# Third subplot
ax4.grid(True)
line44, = ax4.plot(time_PIOT_long, data_PIOT_long['Plug1_measured_P'], label='Power to EV1', color=EVSEcolors[0])
ax4.set_ylabel('EV1 Consumed\n Power [kW]', fontsize=DefFontSize * 1, color = EVSEcolors[0])
ax4.set_xlabel('Time [h:m]', fontsize=DefFontSize * 1)
ax4.legend(loc="best")
ax4.set_ylim(6, 10.5)
ax55 = ax4.twinx()
line55, = ax55.plot(time_PIOT_long, data_PIOT_long['Plug2_measured_P'], color=EVSEcolors[1], label='Power to EV2')
ax55.set_ylabel('EV2 Consumed\nPower [kW]', fontsize=DefFontSize * 1, color = EVSEcolors[1])
ax4.legend([line44, line55], [line44.get_label(), line55.get_label()], loc="best")
ax55.set_ylim(6, 10.5)

# Show x-ticks only on the bottom plot
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('./graphs/Combined_All_Plots.pdf')
#plt.show()


fig = plt.figure()
plt.plot(time_CH, data_CH['priority_VA0'])
plt.plot(time_CH, data_CH['priority_VA1'])
# Adding titles and labels
plt.title('Data over Time')
plt.xlabel('Time (time_CH)')
plt.ylabel('Data (data_CH)')

# Show the plot
plt.show()


'''
plt.plot(time_CH, data_CH['priority_VA0'])
plt.plot(time_CH, data_CH['priority_VA1'])
# Adding titles and labels
plt.title('Data over Time')
plt.xlabel('Time (time_CH)')
plt.ylabel('Data (data_CH)')

# Show the plot
plt.show()



fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.grid(True)
line1, = ax1.plot(time_PIOT_long, data_PIOT_long['REG_FREQUENCY'], label='Measured frequency', color=EVSEcolors[0])
ax1.set_ylim(49.97, 50.055) 
ax1.set_ylabel('Frequency', fontsize=DefFontSize * 1,color=EVSEcolors[0])
ax1.set_xlabel('Time [s]', fontsize=DefFontSize * 1)
xticks = ax1.get_xticks()
#xticklabels = ax1.get_xticklabels()
#ax1.set_xticks(xticks)
#ax1.set_xticklabels(xticklabels, rotation=90)
ax2 = ax1.twinx()
line2, = ax2.plot(time_PIOT_long, -data_PIOT_long['REG_LINE_ACP_1'] -data_PIOT_long['REG_LINE_ACP_2'] -data_PIOT_long['REG_LINE_ACP_3'], color=EVSEcolors[1], label= 'Cluster Power')
ax2.set_ylim(13.5, 15.75)
ax2.set_ylabel('Cluster power [kW]', fontsize=DefFontSize * 1,color=EVSEcolors[1])

# Combine lines and labels from both axes for the legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]

# Set the unified legend on ax1 or ax2
ax1.legend(lines, labels, loc="best")

#date_format = mdates.DateFormatter('%H:%M')
#ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
#ax2.xaxis.set_major_formatter(date_format)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.savefig('./graphs/Power_and_frequency_priority', format='png')


# PRIORITY

fig, ax3 = plt.subplots(figsize=(15, 5))
plt.grid(True)
ax3.plot(time_PIOT_long, data_PIOT_long['Plug1_priority_broadcast'], label='Priority of EV1', color=EVSEcolors[0])
ax3.plot(time_PIOT_long, data_PIOT_long['Plug2_priority_broadcast'], label='Priority of EV1', color=EVSEcolors[1])
ax3.set_ylabel('Priority of the EV',fontsize=DefFontSize * 1)
#ax3.set_xlabel('Time [h:m]',fontsize=DefFontSize * 1)
#date_format = mdates.DateFormatter('%H:%M')
#ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
#ax3.xaxis.set_major_formatter(date_format)
ax3.legend(loc="best")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.savefig('./graphs/Priorities_priority', format='png')

# POWER

fig, ax4 = plt.subplots(figsize=(15, 5))
ax4.grid(True)
ax4.plot(time_PIOT_long, data_PIOT_long['Plug1_measured_P'], label='Power to EV1', color=EVSEcolors[0])
ax4.set_ylabel('Consumed EV Power [kW]',fontsize=DefFontSize * 1)
ax4.set_xlabel('Time [h:m]',fontsize=DefFontSize * 1)
ax4.legend(loc="best")
ax4.set_ylim(4.5, 6.3)
#date_format = mdates.DateFormatter('%H:%M')
#ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
#ax4.xaxis.set_major_formatter(date_format)
#plt.xticks(rotation=90)
ax55 = ax4.twinx()
line55, = ax55.plot(time_PIOT_long, data_PIOT_long['Plug2_measured_P'], color=EVSEcolors[1], label= 'Cluster Power')
ax55.set_ylim(8.3, 8.7)

plt.tight_layout()
plt.savefig('./graphs/Power_to_EVs_priority', format='png')
plt.show()
'''
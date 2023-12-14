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
data_PIOT = pd.read_csv('./Data/prova1_exp_20231201.csv')

t_start_long = data_PIOT['timestamp'].iloc[-1100]
t_end_long = data_PIOT['timestamp'].iloc[-500]
data_PIOT_long = data_PIOT[(data_PIOT['timestamp'] >= t_start_long) & (data_PIOT['timestamp'] <= t_end_long)]
time_PIOT_long = data_PIOT_long['timestamp']
time_PIOT_long= pd.to_datetime(time_PIOT_long, unit='ms')

t_start_short = data_PIOT['timestamp'].iloc[-1100]
t_end_short = data_PIOT['timestamp'].iloc[-500]
data_PIOT_short = data_PIOT[(data_PIOT['timestamp'] >= t_start_short) & (data_PIOT['timestamp'] <= t_end_short)]
time_PIOT_short = data_PIOT_short['timestamp']
time_PIOT_short= pd.to_datetime(time_PIOT_short, unit='ms')


plug_state = np.zeros((6, len(data_PIOT['Plug1_status'])))



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
fig, ax5 = plt.subplots(figsize=(15, 5))
ax5.grid(True)
line1, = ax5.plot(time_PIOT_short, data_PIOT_short['REG_FREQUENCY'], label='Measured frequency', color=EVSEcolors[0])
ax5.set_ylim(49.96, 50.06) 
#ax5.set_ylabel('Frequency', fontsize=DefFontSize * 1,color=EVSEcolors[0])
#ax5.set_xlabel('Time [h:m]', fontsize=DefFontSize * 1)
#xticks = ax5.get_xticks()
#xticklabels = ax5.get_xticklabels()
#ax5.set_xticks(xticks)
#ax5.set_xticklabels(xticklabels, rotation=90)
ax6 = ax5.twinx()
line2, = ax6.plot(time_PIOT_short, -data_PIOT_short['REG_LINE_ACP_1'] -data_PIOT_short['REG_LINE_ACP_2'] -data_PIOT_short['REG_LINE_ACP_3'], color=EVSEcolors[1], label= 'Cluster Power')
ax6.set_ylim(13, 15.6)
ax6.set_ylabel('Cluster power [kW]', fontsize=DefFontSize * 1, color=EVSEcolors[1])

# Combine lines and labels from both axes for the legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]

# Set the unified legend on ax5 or ax6
ax5.legend(lines, labels, loc="best")

#date_format = mdates.DateFormatter('%M:%S')
#ax6.xaxis.set_major_locator(mdates.SecondLocator(interval=7))
#ax6.xaxis.set_major_formatter(date_format)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
plt.savefig('./graphs/Power_and_frequency_short', format='png')
'''
"""
for i in range(len(data_PIOT['Plug1_status'])):
    if data_PIOT['Plug2_status'][i] == 'Idle':
        plug_state[2, i] = 0
    elif data_PIOT['Plug2_status'][i] == 'Connected':
        plug_state[2, i] = 0.5
    elif data_PIOT['Plug2_status'][i] == 'Charging':
        plug_state[2, i] = 1
    else:
        plug_state[2, i] = -0.5
    if data_PIOT['Plug3_status'][i] == 'Idle':
        plug_state[3, i] = 0
    elif data_PIOT['Plug3_status'][i] == 'Connected':
        plug_state[3, i] = 0.5
    elif data_PIOT['Plug3_status'][i] == 'Charging':
        plug_state[3, i] = 1
    else:
        plug_state[3, i] = -0.5
"""
# PLUG STATE
'''
fig1, ax = plt.subplots()
custom_y_labels = ['Error','Idle','Connected','Charging']
custom_y_positions = [-.5,0,.5,1]
plt.plot(time_PIOT, plug_state[2, :], label='EV 2', color=EVSEcolors[0])
plt.plot(time_PIOT, plug_state[3, :], label='EV 3', color=EVSEcolors[1])
plt.ylim(-0.55, 1.1)

ax.set_yticks(custom_y_positions)
ax.set_yticklabels(custom_y_labels)
plt.ylabel(r'$s_{EVSE}$', fontsize=DefFontSize * 1)
plt.xlabel(r'time $[s]$', fontsize=DefFontSize * 1)
plt.legend(loc="best")
#plt.savefig('DEMOoutput\\state.pdf', format='pdf')
'''
# FREQUENCY
'''
fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.grid(True)
ax1.plot(time_PIOT, data_PIOT['REG_FREQUENCY'], label='frequency', color=EVSEcolors[0])
ax1.set_ylim(49.8, 50.2) 
ax1.set_ylabel('Frequency', fontsize=DefFontSize * 1)
ax1.set_xlabel('Time [s]', fontsize=DefFontSize * 1)
ax2 = ax1.twinx()  
ax2.set_ylabel('Cluster power [kW]', color=EVSEcolors[1])
ax2.plot(time_PIOT, -data_PIOT['REG_LINE_ACP_1'] -data_PIOT['REG_LINE_ACP_2'] -data_PIOT['REG_LINE_ACP_3'], color=EVSEcolors[1], label= 'Cluster Power')
ax2.set_ylim(10, 20)
plt.legend(loc="best")
date_format = mdates.DateFormatter('%H:%M')
ax2.xaxis.set_major_formatter(date_format)
plt.xticks(rotation=90)
#plt.savefig('DEMOoutput\\frequency.pdf', format='pdf')
plt.show()
'''
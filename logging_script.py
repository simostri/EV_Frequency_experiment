# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:55:02 2024

@author: sistri
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime

def eval_datetime(dt_str):
    try:
        return eval(dt_str)
    except (NameError, SyntaxError):
        return pd.NaT  # Re
    
# Set layout constants
plt.close('all')
plt.rcParams['font.size'] = 20 
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
DefFontSize = 20
EVSEcolors = ['blue', 'red', 'green']
t_start_test = pd.to_datetime('2024-01-16 11:27:50')
t_end_test = pd.to_datetime('2024-01-16 11:50:05')
P_FL = 19
P_bid=3
k = 2*P_bid/0.2
#t_start_chargers = chargers_log['timestamp'].iloc[-1000]
#t_end_chargers = chargers_log['timestamp'].iloc[-1]



SOC = pd.read_excel('./Data/test_2024_01_16/SOC_development_epoch.xlsx')
SOC['time']= pd.to_datetime(SOC['time'], unit='s')


t_start_hd = pd.to_datetime('2024-01-16 10:43:01')
t_end_hd = pd.to_datetime('2024-01-16 10:46:59')

ACP_SUM_HD = pd.read_csv('./Data/test_2024_01_16/ACPSUM_HD.csv',skiprows=2,names=['Time', 'ACP SUM'])
ACP_SUM_HD ['Time']= pd.to_datetime(ACP_SUM_HD ['Time'], unit='ms')
freq_SUM_HD = pd.read_csv('./Data/test_2024_01_16/FREQSUM_HD.csv',skiprows=2,names=['Time', 'frequency'])
freq_SUM_HD['Time']= pd.to_datetime(freq_SUM_HD['Time'], unit='ms')
freq_SUM_HD['P_f_ref'] = (P_FL - P_bid) + k*(freq_SUM_HD['frequency']- 50)
ACP_SUM_HD = ACP_SUM_HD[(ACP_SUM_HD['Time'] >= t_start_hd) & (ACP_SUM_HD['Time'] <= t_end_hd)]
freq_SUM_HD = freq_SUM_HD[(freq_SUM_HD['Time'] >= t_start_hd) & (freq_SUM_HD['Time'] <= t_end_hd)]
time = freq_SUM_HD['Time']
plt.figure(figsize=(12, 6))
plt.plot(ACP_SUM_HD['Time'], -ACP_SUM_HD['ACP SUM'], label='ACP SUM', color='blue')
plt.plot(freq_SUM_HD['Time'], freq_SUM_HD['P_f_ref'], label='Frequency', color='red')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('ACP SUM and Frequency Over Time')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()

ACP_centered = -ACP_SUM_HD['ACP SUM']- np.mean(-ACP_SUM_HD['ACP SUM'])
freq_centered = freq_SUM_HD['P_f_ref'] - np.mean(freq_SUM_HD['P_f_ref'])
time_delta = (time.diff().dt.total_seconds().mean())
# Compute the cross-correlation
cross_correlation = np.correlate(ACP_centered, freq_centered, mode='full')
# Normalize the cross-correlation
normalization_factor = np.std(ACP_centered) * np.std(freq_centered) * len(ACP_centered)
normalized_cross_correlation = cross_correlation / normalization_factor

lags = np.arange(-(len(time) - 1), len(time))
time_intervals = lags * time_delta
plt.figure(figsize=(10, 6))
plt.plot(time_intervals, normalized_cross_correlation)
plt.xlim(0, 1000 * time_delta) 
plt.title('Cross_corr_normalized')
plt.xlabel('Lags')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.savefig('./graphs/Correlation_HD.pdf')

lag_at_max_corr = time_intervals[np.argmax(normalized_cross_correlation)]
print("The lag is:", lag_at_max_corr, "time units")


# Define the file path (you'll need to set this to your file's path)
prova1 = pd.read_csv('./Data/test_2024_01_16/prova1_20240116.csv')
prova1['timestamp']= pd.to_datetime(prova1['timestamp'], unit='ms')
prova1 = prova1[(prova1['timestamp'] >= t_start_test) & (prova1['timestamp'] <= t_end_test)]
prova1['P_f_ref'] = (P_FL - P_bid) + k*(prova1['REG_FREQUENCY']- 50)
#prova1['P_f_ref']= max( min(prova1['P_f_ref'], P_FL), P_FL-2*P_bid)

##### Preparing chargers_log -----------------------------------------------------------------
chargers_log = pd.read_csv('./Data/test_2024_01_16/Log_1705405987_CH_acdc-352656103199330.csv')
chargers_log = chargers_log.drop('Unnamed: 0', axis=1)
chargers_log['timestamp'] = pd.to_datetime(chargers_log['timestamp'], unit='s')
chargers_log = chargers_log[(chargers_log['timestamp'] >= t_start_test) & (chargers_log['timestamp'] <= t_end_test)]
# Preparing CA log -----------------------------------------------------------------
CA_log = pd.read_csv('./Data/test_2024_01_16/Log_1705405998_CA.csv')
CA_log = CA_log.drop('Unnamed: 0', axis=1)
CA_log['timestamp'] = pd.to_datetime(CA_log['timestamp'], unit='s')
CA_log = CA_log[(CA_log['timestamp'] >= t_start_test) & (CA_log['timestamp'] <= t_end_test)]
# Preparing Local log -----------------------------------------------------------------
Local_log = pd.read_csv('./Data/test_2024_01_16/preprocessed_pwm_data.csv')
Local_log ['timestamp'] = pd.to_datetime(Local_log ['timestamp'], unit='ns')
Local_log = Local_log [(Local_log ['timestamp'] >= t_start_test) & (Local_log ['timestamp'] <= t_end_test)]
Local_log['Freq'] = (Local_log['Freq'] / 1000) * 50
"""for i in range(1, len(Local_log['Freq'])):
    if Local_log['Freq'].iloc[i] <= 49.9 or Local_log['Freq'].iloc[i] >= 50.1:
            # Replace it with the previous value
            Local_log['Freq'].iloc[i] = Local_log['Freq'].iloc[i - 1]
"""



for i in range(1, len(Local_log['DutyCycle'])):
    # Check if the value is outside the desired range
    if Local_log['DutyCycle'].iloc[i]*100*0.6*0.23*3 < 8 or Local_log['DutyCycle'].iloc[i]*100*0.6*0.23*3 > 11:
        # Replace it with the previous value
        Local_log['DutyCycle'].iloc[i] = Local_log['DutyCycle'].iloc[i - 1]
# Preparing merge between Local log and prova 1 -----------------------------------------------------------------
Local_log_subset = Local_log[['timestamp', 'DutyCycle','idiff']]
prova1_subset = prova1[['timestamp', 'REG_PHASE_VOL_1','REG_PHASE_VOL_2','REG_PHASE_VOL_3']]
merged_df = pd.merge_asof(Local_log_subset.sort_values('timestamp'), prova1_subset.sort_values('timestamp'), on='timestamp', direction='nearest')
merged_df['EV1_power'] = merged_df['DutyCycle']*100*0.6* merged_df['REG_PHASE_VOL_1']/1000 + \
 merged_df['DutyCycle']*100*0.6* merged_df['REG_PHASE_VOL_2']/1000 + \
 merged_df['DutyCycle']*100*0.6* merged_df['REG_PHASE_VOL_3']/1000
merged_df['Local_P_meas_tot'] = merged_df['idiff'] *\
    (merged_df['REG_PHASE_VOL_1']+merged_df['REG_PHASE_VOL_2']+merged_df['REG_PHASE_VOL_3'])/1000



#Local_log = Local_log.sort_values('timestamp')
#time = Local_log['timestamp']
#ACP = Local_log['idiff']*0.23*3
#freq = (P_FL - P_bid) + k*(Local_log['Freq']- 50)

ACP = -prova1['REG_LINE_ACP_1'] - prova1['REG_LINE_ACP_2'] - prova1['REG_LINE_ACP_3']
time = prova1['timestamp']
freq = prova1['P_f_ref']
ACP_centered = ACP - np.mean(ACP)
freq_centered = freq - np.mean(freq)
time_delta = (time.diff().dt.total_seconds().mean())
# Compute the cross-correlation
cross_correlation = np.correlate(ACP_centered, freq_centered, mode='full')
# Normalize the cross-correlation
normalization_factor = np.std(ACP_centered) * np.std(freq_centered) * len(ACP_centered)
normalized_cross_correlation = cross_correlation / normalization_factor

lags = np.arange(-(len(freq) - 1), len(freq))
time_intervals = lags * time_delta
plt.figure(figsize=(10, 6))
plt.plot(time_intervals, normalized_cross_correlation)
plt.xlim(0, 1000 * time_delta) 
plt.title('Cross_corr_normalized')
plt.xlabel('Lags')
plt.ylabel('Cross-correlation')
plt.grid(True)
plt.savefig('./graphs/Cross_corr_normalized.pdf')

lag_at_max_corr = time_intervals[np.argmax(normalized_cross_correlation)]
print("The lag is:", lag_at_max_corr, "time units")


plt.figure(figsize=(10, 6))
plt.plot(time, ACP)
plt.plot(time, freq)
plt.title('Cross_corr_normalized')
plt.xlabel('Lags')
plt.ylabel('Cross-correlation')
plt.grid(True)


fig, (ax1, ax3, ax4) = plt.subplots(nrows=3, ncols=1, figsize=(15, 12), sharex=True)
# First subplot ---------------------------------------------------------------
ax1.grid(True)
line1, = ax1.plot(prova1['timestamp'], prova1['REG_FREQUENCY'], label='Measured frequency', color=EVSEcolors[0])
ax1.set_ylim(49.89, 50.07) 
ax1.set_ylabel('Frequency [Hz]', fontsize=DefFontSize * 1, color=EVSEcolors[0])
ax2 = ax1.twinx()
line2, = ax2.plot(prova1['timestamp'], -prova1['REG_LINE_ACP_1'] - prova1['REG_LINE_ACP_2'] - prova1['REG_LINE_ACP_3'], color=EVSEcolors[1], label='Cluster Power')
line3, = ax2.plot(prova1['timestamp'], prova1['P_f_ref'], color=EVSEcolors[2], label='P_expected')
ax2.set_ylim(12.700000000000017, 18.10000000000001)
ax2.set_ylabel('Cluster power [kW]', fontsize=DefFontSize * 1, color=EVSEcolors[1])
ax1.legend([line1, line2,line3], [line1.get_label(), line2.get_label(), line3.get_label()], loc="best")
# Second subplot ---------------------------------------------------------------
ax3.grid(True)
ax3.plot(chargers_log['timestamp'], chargers_log['priority_VA0'], label='Priority of EV1', color=EVSEcolors[0])
ax3.plot(chargers_log['timestamp'], chargers_log['priority_VA1'], label='Priority of EV2', color=EVSEcolors[1])
ax3.set_ylabel('Priority of the EV', fontsize=DefFontSize * 1)
ax3.legend(loc="best")
# Third subplot ---------------------------------------------------------------
ax4.grid(True)
line44, = ax4.plot(prova1['timestamp'], prova1['Plug1_measured_P'], label='Power to EV1', color=EVSEcolors[0])
ax4.set_ylabel('EV Consumed\n Power [kW]', fontsize=DefFontSize * 1, color = EVSEcolors[0])
ax4.set_xlabel('Time [h:m]', fontsize=DefFontSize * 1)
ax4.legend(loc="best")
ax4.set_ylim(3, 10.5)
line55, = ax4.plot(prova1['timestamp'], prova1['Plug2_measured_P'], color=EVSEcolors[1], label='Power to EV2')

# Show x-ticks only on the bottom plot
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('./graphs/Combined_All_Plots.pdf')
#plt.show()



fig = plt.figure()
plt.plot(chargers_log['timestamp'], chargers_log['P_ref_CA'], label='P_ref_CA on charger')
plt.plot(CA_log['timestamp'], CA_log['PCC_ref_A'] + CA_log['PCC_ref_B'] + CA_log['PCC_ref_C'], label='P_ref_CA on CA')
plt.plot(merged_df['timestamp'], merged_df['EV1_power'], label='Local EV1 P_meas')
plt.plot(chargers_log['timestamp'], chargers_log['P_meas_pcc'], label='P_meas_pcc')
plt.plot(merged_df['timestamp'], merged_df['Local_P_meas_tot'], label='Local_P_meas_tot')
plt.plot(chargers_log['timestamp'], (chargers_log['P_ref_VA0']+chargers_log['P_ref_VA1'])*3, label='VAs_tot_ref')
plt.plot(chargers_log['timestamp'], chargers_log['P_ref_VA0']*3, label='P_ref_VA0 from VA')
plt.plot(chargers_log['timestamp'], chargers_log['P_ref_VA1']*3, label='P_ref_VA1 from VA')
plt.plot(prova1['timestamp'], prova1['Plug1_measured_P'], label='Pmeas EV1 from ch')
plt.plot(prova1['timestamp'], prova1['Plug2_measured_P'], label='Pmeas EV2 from ch')
plt.plot(prova1['timestamp'], prova1['Plug1_measured_P']+prova1['Plug2_measured_P'], label='Pmeas EV1+EV2 from ch')
#plt.ylim(0,25)
# Adding titles and labels
plt.legend(loc='upper left')
plt.title('Check delay between chargers_log and CA log for P_ref_CA')
plt.xlabel('Time (chargers_log)')
plt.ylabel('Data (data_CH)')
plt.grid(True)

plt.gcf().autofmt_xdate()

# Show the plot
plt.show()


fig = plt.figure()
plt.plot(SOC['time'], SOC['Zoe_Simone'], label='Zoe 1')
plt.plot(SOC['time'], SOC['Zoe_DTU'], label='Zoe 2')
#plt.ylim(0,25)
# Adding titles and labels
plt.legend(loc='best')
plt.title('SOC development of the cars')
plt.xlabel('Time')
plt.ylabel('SOC')
plt.grid(True)

plt.gcf().autofmt_xdate()

plt.savefig('./graphs/SOC.pdf')
plt.show()

"""
fig = plt.figure()
plt.plot(Local_log['timestamp'], Local_log['Freq'], label='Local_log_freq')
plt.plot(prova1['timestamp'], prova1['REG_FREQUENCY'], label='PIOT freq')
#plt.ylim(0,25)
# Adding titles and labels
plt.legend(loc='best')
plt.title('Check delay frequency')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)

plt.gcf().autofmt_xdate()

# Show the plot
plt.show()
"""
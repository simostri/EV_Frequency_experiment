# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:16:39 2023

@author: sistri
"""

import datetime
import socket
import ntplib

def sync_time_with_ntp(server="pool.ntp.org"):
    client = ntplib.NTPClient()
    response = client.request(server, version=3)
    return datetime.datetime.utcfromtimestamp(response.tx_time)

def check_time_difference():
    local_time = datetime.datetime.utcnow()
    ntp_time = sync_time_with_ntp()
    return local_time - ntp_time, ntp_time

# Example Usage
time_diff, ntp_time = check_time_difference()
print(f"Time difference from NTP server: {time_diff}")
print(datetime.datetime.utcnow()-time_diff)
print(ntp_time)

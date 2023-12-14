# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:25:58 2023

@author: krisse
"""

import serial
import time
import datetime
import numpy as np
import csv
import ntplib

def sync_time_with_ntp(server="pool.ntp.org"):
    client = ntplib.NTPClient()
    response = client.request(server, version=3)
    return datetime.datetime.utcfromtimestamp(response.tx_time)

def check_time_difference():
    local_time = datetime.datetime.utcnow()
    ntp_time = sync_time_with_ntp()
    return local_time - ntp_time


print(serial.__version__)
# 3.4
ser = serial.Serial(
    port='COM3',
    baudrate=19200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=2
    # timeout = 1
)

cstatedict = {14: "State A", 12: "State B", 8: "State C", 0: "State D"}

state = 'wait_sofesc'
numframes = 0
framenum = 0
temp = 0
recstring = []
textbuffer = []
done = False

filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file = open('PWM_meas'+filename + '.csv', 'w', newline='')
writer = csv.writer(file)

time_diff = check_time_difference()

while True:
    if ser.in_waiting > 0:
        x = ord(ser.read())
        if state == 'wait_sofesc':
            if x == 0x7e:
                state = 'wait_sof'
        elif state == 'wait_sof':
            if x == 0xaa:
                recstring = []
                state = 'get_body'
            else:
                state = 'wait_sofesc'
        elif state == 'get_body':
            if x == 0x7e:
                state = 'rec_escape'
            else:
                recstring.append(x)
        elif state == 'rec_escape':
            if x == 0x55:
                done = True
                state = 'wait_sofesc'
            elif x == 0xaa:
                state = 'wait_sofesc'
            else:
                recstring.append(x)
                state = 'get_body'

        if done:
            crc_received = recstring[len(recstring) - 1]
            crc = 0x00
            for a in recstring[:-1]:
                extract = a
                for tempI in range(8, 0, -1):
                    sum = (crc ^ extract) & 0x01
                    crc >>= 1
                    if sum:
                        crc ^= 0x8C
                    extract >>= 1

            print("CRC Expected " + str(crc) + ", received " + str(crc_received))

            if crc != crc_received:
                print("CRC Error! Expected " + str(crc) + ", received " + str(crc_received))
            if len(recstring) != 10:
                print("Error! Expected message length of 10, received " + str(len(recstring)))

            else:
                seqno = (256 * recstring[0] + recstring[1])

                # Get current time in milliseconds
                current_time_milliseconds = datetime.datetime.utcnow()-time_diff

                print("seqno =", seqno)
                print("Time (ms) =", current_time_milliseconds)  # Print the current time in milliseconds

                pwmlen = (256 * recstring[2] + recstring[3])
                print("pwmlen =", pwmlen * 40)
                pwmduty = (256 * recstring[4] + recstring[5])
                print("pwmduty =", pwmduty * 40)
                idiff = (256 * recstring[6] + recstring[7])
                chgstate = (recstring[8] & 0x0e)
                print("idiff =", idiff / (64 * 19.13))
                print("Freq =", (2e6 / pwmlen))
                print("Duty cycle =", (pwmduty / pwmlen))
                print("State =", cstatedict[chgstate])

                data = [["seqno=", seqno],
                        ["Time (ms)=", current_time_milliseconds],
                        ["pwmlen=", pwmlen * 40],
                        ["pwmduty=", pwmduty * 40],
                        ["idiff", idiff / (64 * 19.13)],
                        ["Freq=", (2e6 / pwmlen)],
                        ["Duty cycle=", (pwmduty / pwmlen)],
                        ["State=", cstatedict[chgstate]]]
                writer.writerow(data)
            done = False

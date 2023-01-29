# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 08:15:40 2023

@author: scott.mckenzie
"""

import matplotlib.pyplot as plt
import numpy as np

#import scipy.signal
from scipy import constants
#from chirp import chirp


def nearest_base2(value):
    value -= 1
    value = value | (value >> 1)
    value = value | (value >> 2)
    value = value | (value >> 4)
    value = value | (value >> 8)
    value = value | (value >> 16)
    value += 1
    return value

#only good for even carrier frequencies at the moment
def barker(b, tau, chip_tau, fc, Fs):
    if(1/fc > chip_tau):
        raise Exception("center frequency period is greater than chip period")
    if(chip_tau >= tau):
        raise Exception("chip period is greater than or equal to pulse period")
    #if(tau%chip_tau != 0):
    #    raise Exception("chip tau is not a multiple of tau")                   #just make sure tau modulo chip tau is zero for right now. seems to have a problem for values under zero. Could be a floating point error
    nChipSamps = int(Fs * chip_tau)
    nPulseSamps = int(Fs * tau)
    tChip = np.linspace(0, chip_tau, nChipSamps)        
    pChip = np.zeros(nChipSamps)
    nChip = np.zeros(nChipSamps)
    pChip = np.sin(constants.pi * tChip * fc)
    nChip = np.sin(constants.pi * tChip * fc + constants.pi)
    signal = np.zeros(nPulseSamps * 4)
    b_idx = 0
    for chipIdx in range(0, nPulseSamps, nChipSamps - 1):
        print(chipIdx)
        if(b[b_idx] > 0):
            signal[chipIdx:chipIdx + nChipSamps] = pChip
        elif(b[b_idx] < 0):
            signal[chipIdx:chipIdx + nChipSamps] = nChip
        b_idx += 1
        if(b_idx >= len(b)):
            break
    plt.figure(10)
    plt.plot(signal)
    
    

def problem_one():
    b = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
    tau = 13e-6
    sig_len = tau / .25
    chip_len = 50
    samps_per_pulse = len(b) * chip_len
    signal = np.zeros(samps_per_pulse * 4)
    b_idx = 0
    for idx in range(0, int(len(signal) * 0.25), chip_len):
        if(b[b_idx] == 1):
            for sub_idx in range(0, chip_len, 1):
                signal[idx + sub_idx] = 1.0
        elif(b[b_idx] == -1):
            for sub_idx in range(0, chip_len, 1):
                signal[idx + sub_idx] = -1.0
        b_idx +=1
    t = np.linspace(0.0, 4 * tau, len(signal))
    plt.figure(1)
    plt.plot(t,signal)
    matched_sig = np.flip(signal[0:samps_per_pulse])
    matched_out = np.convolve(signal, matched_sig)
    plt.figure(2)
    plt.plot(matched_sig)
    plt.figure(3)
    plt.plot(20.0*np.log10(np.abs(np.fft.fftshift(np.fft.fft(matched_out)))))
    #plt.plot(np.abs(np.fft.fftshift(np.fft.fft(matched_out))))
    #plt.ylim(20, 100)
    #plt.xlim(2183, 3015)
    t_m = np.linspace(0.0, 4 * tau, len(matched_out))
    plt.figure(4)
    plt.plot(t_m, matched_out)
    barker(b, 13e-6, 1e-6, 4e6, 100e6)


def problem_two():
    B = 25e6
    tau = 1e-6
    Fs = 5 * B
    #Fc = B / 2
    #nSamps = int(Fs * tau)
    #nSamps = nearest_base2(int(Fs * tau))
    nSamps = int(((tau/2) - (-tau/2)) * Fs)
    t = np.linspace(-tau/2, tau/2, nSamps)
    #t = np.linspace(0.0, tau, nSamps)
    #f = np.linspace(Fc - B / 2, Fc + B /2, len(t))
    signal = np.zeros(nSamps*4)
    signal[0:len(t)] = np.exp(1j*constants.pi*(B/tau)*((t-tau)**2))
    #signal[0:len(t)] = np.sin(constants.pi * t * f)
    #signal[0:len(t)] += np.random.randn(len(signal[0:len(t)])) * 10**(-60/20.0) 
    matched_sig = np.conjugate(np.flip(signal))
    #matched_sig = np.conjugate(np.flip(signal[0:len(t)]))
    filter_output = np.convolve(signal, matched_sig)
    plt.figure(5)
    plt.plot(signal)
    #plt.figure(6)
    #plt.plot(np.abs(np.fft.rfft(signal)))
    #plt.plot(20*np.log10(np.abs(np.fft.rfft(signal))))
    window_1 = np.sin(constants.pi * t / tau)
    wind_sig_1 = np.zeros(nSamps*4)
    wind_sig_1[0:len(t)] = np.multiply(signal[0:len(t)], window_1)
    plt.figure(7)
    plt.plot(wind_sig_1)
    plt.figure(6)
    #plt.plot(np.abs(np.fft.rfft(wind_sig_1)))
    #plt.plot(20.0*np.log10(np.abs(np.fft.rfft(wind_sig_1))))[0:len(t)])
    sig1 = np.convolve(wind_sig_1, matched_sig)
    plt.plot(abs(sig1)/max(abs(sig1)))
    window_2 = np.sin(constants.pi * t / tau) ** 2
    wind_sig_2 = np.zeros(nSamps*4)
    wind_sig_2[0:len(t)] = np.multiply(signal[0:len(t)], window_2)
    plt.figure(8)
    plt.plot(wind_sig_2)
    plt.figure(11)
    sig2 = np.convolve(wind_sig_2,matched_sig)
    plt.plot(abs(sig2)/max(abs(sig2)))
    #plt.plot(np.abs(np.fft.rfft(wind_sig_2)))
    #plt.plot(20.0*np.log10(np.abs(np.fft.rfft(wind_sig_2))))
    plt.figure(9)
    plt.plot(abs(filter_output)/max(abs(filter_output)))




if __name__ == "__main__":
    problem_one()
    problem_two()
    #problem_three()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:54:29 2023

@author: slmckenzie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, c

def range_resolution_pulse(tau):
    return c * tau / 2.0

def range_resolution_lfm(bw):
    return c / (2.0 * bw)

def barker(b, tau, Fs, Fc, Fd = 0.0, amp = -1):
    chip_tau = tau / len(b)
    nPulseSamps = int(tau * Fs)
    nChipSamps = int(chip_tau * Fs)
    t = np.linspace(0, tau, nPulseSamps)
    expanded_barker = np.repeat(b, nChipSamps, axis = 0)
    modulation = np.append(2 * expanded_barker - 1, np.zeros(nPulseSamps * 3))
    coded_phase = amp * np.sin(2 * pi * t * (Fc + Fd) + pi * expanded_barker)
    signal = np.append(coded_phase, np.zeros(nPulseSamps * 3))
    return modulation, signal

def problem_one():
    b = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    tau = 13e-6
    Fc = 2e6
    Fs = 200e6
    Fd = 30e3
    
    barkPulse, signal = barker(b, tau, Fs, Fc)
    throwAway, dop_signal = barker(b, tau, Fs, Fc, Fd)
    matched_filt = np.flip(signal)
    output = np.convolve(signal, matched_filt)    
    matched_pulse = np.flip(barkPulse)
    pulse_output = np.convolve(barkPulse, matched_pulse)
    mismatched = np.convolve(dop_signal, matched_filt)
    t = np.linspace(0.0, 4 * tau, len(signal))
    t_m = np.linspace(0.0, 4 * tau, len(output))
    
    end_t = int(tau * Fs)
    ax = plt.subplot(2, 1, 1)
    ax.plot(t[0:end_t] / tau, signal[0:end_t], 
            t[0:end_t] / tau, barkPulse[0:end_t], linewidth = 4)
    ax.plot(t[0:end_t] / tau, dop_signal[0:end_t], linestyle = ':', color = 'k')
    ax.set_xlim(0.0, t[end_t] / tau)
    ax.grid(True)
    plt.xlabel("Normalized Time [t / tau]")
    plt.title("Barker Coded signal")
    # plt.ylabel("Barker Coded signal")
    
    
    ax = plt.subplot(2, 1, 2)
    ax.plot(t_m /tau, abs(output)/max(abs(pulse_output)), 
            t_m / tau, abs(pulse_output)/max(abs(pulse_output)), linewidth = 4)
    ax.plot(t_m / tau, abs(mismatched)/max(abs(pulse_output)), linestyle = ':', color = 'k')
    ax.set_xlim(1.0, 3.0)
    ax.grid(True)
    plt.xlabel("Normalized Time [t / tau]")
    plt.title("Matched Filter output")
    # plt.ylabel("Matched Filter output")
    
    print("Range resolution of pulse with tau = {0:.6f} seconds equals = {1:.3f} meters".format(1e-6, range_resolution_pulse(1e-6)))
    print("Range resolution of modulated pulse with tau = {0:.6f} seconds, and chip width of {1:.6f} seconds,  equals = {2:.3f} meters".format(13e-6, 1e-6, range_resolution_pulse(1e-6)))
    
def problem_two():
    tau = 1e-6 
    beta = 25e6 
    Fs = 20* beta 
    
    t = np.linspace(-tau / 2, tau / 2, int((tau / 2 - (-tau / 2)) * Fs)) 
    t_val = np.linspace(-tau / 2, tau / 2, len(t) * 4 - 1) 
    t_wind = np.linspace(0, tau, int((tau / 2 - (-tau / 2)) * Fs))
    t_plot = np.linspace(0, 2 * tau, int((tau / 2 - (-tau / 2)) * Fs) * 2)
    
    mod_signal = np.exp(1j * pi * (beta / tau) * ((t - tau)**2)) 
    # mod_signal = np.exp(1j * pi * (beta / tau) * ((t)**2)) 
    
    modSig_in = np.append(mod_signal, np.zeros(len(t))) 
    modSig_conj = np.conjugate(np.flip(modSig_in)) 
    modSig_out = np.convolve(modSig_in, modSig_conj) 
    
    window_1 = np.sin(pi * (t_wind / tau))
    wind_sig_1 = np.zeros((len(t) * 2))
    wind_sig_1[0:len(t)] = np.multiply(mod_signal[0:len(t)], window_1)
    wind1_sig_filt = np.conjugate(np.flip(wind_sig_1))
    wind1_output = np.convolve(modSig_in, wind1_sig_filt)
    
    window_2 = np.sin(np.sin(pi * (t_wind / tau)))
    wind_sig_2 = np.zeros(len(t) * 2)
    wind_sig_2[0:len(t)] = np.multiply(mod_signal[0:len(t)], window_2)
    wind2_sig_filt = np.conjugate(np.flip(wind_sig_2))
    wind2_output = np.convolve(modSig_in, wind2_sig_filt)
    
    plt.figure()
    ax = plt.subplot(3, 1, 1)
    ax.plot(t_plot, modSig_in, linewidth = 3)
    plt.xlim(0.0, t_plot[len(t_plot) - 1])
    ax.grid(True)
    ax = plt.subplot(3, 1, 2)
    ax.plot(t_plot, modSig_conj)
    ax.plot(t_plot, wind1_sig_filt, linewidth = 3)
    ax.plot(t_plot, wind2_sig_filt, linewidth = 3)
    plt.xlim(0.0, t_plot[len(t_plot) - 1])
    ax.grid(True)
    ax = plt.subplot(3, 1, 3)
    plt.plot(t_val/tau, abs(modSig_out)/max(abs(modSig_out)), linewidth = 3) 
    plt.plot(t_val/tau, abs(wind1_output)/max(abs(modSig_out)))
    plt.plot(t_val/tau, abs(wind2_output)/max(abs(modSig_out)))
    plt.xlim(t_val[0]/tau, t_val[len(t_val) - 1]/tau)
    ax.grid(True)

def problem_three():
    beta = 200e6
    tau = 1.5e-6
    
    print("Time badwidth product: {0:.3f}".format(beta * tau))
    print("Range resolution: {0:.4f} meters".format(range_resolution_lfm(beta)))
    print("Minimum sample rate: {}".format(2 * beta))
    print("Number of samples per pulse: {}".format(2 * beta * tau))
    
    

if __name__ == "__main__":
    plt.close("all")
    problem_one()
    problem_two()
    problem_three()
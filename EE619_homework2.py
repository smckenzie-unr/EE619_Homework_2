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

def complex_multipy(complex_vect, scalar_vect):
    if(len(complex_vect) != len(scalar_vect)):
        raise Exception("Vector size do not match")
    new_vect = [0.0 + 1j * 0.0] * len(complex_vect)
    for idx in range(0, len(complex_vect), 1):
        new_vect[idx] = complex(complex_vect[idx].real * scalar_vect[idx],
                                complex_vect[idx].imag * scalar_vect[idx])
    return new_vect

def barker(b, tau, Fs, Fc, Fd = 0.0, amp = -1, dtype = "real"):
    chip_tau = tau / len(b)
    nPulseSamps = int(tau * Fs)
    nChipSamps = int(chip_tau * Fs)
    t = np.linspace(0, tau, nPulseSamps)
    expanded_barker = np.repeat(b, nChipSamps, axis = 0)
    modulation = np.append(2 * expanded_barker - 1, np.zeros(nPulseSamps * 3))
    if(dtype == "real"):
        coded_phase = amp * np.sin(2 * pi * t * (Fc + Fd) + pi * expanded_barker)
        signal = np.append(coded_phase, np.zeros(nPulseSamps * 3))
    elif(dtype == "complex"):
        coded_phase = amp * np.exp(1j * (2 * pi * t * (Fc + Fd) + pi * expanded_barker + pi / 2))
        signal = np.append(coded_phase, np.zeros(nPulseSamps * 3, dtype = complex))
    return modulation, signal

def problem_one(doComplex = False):
    b = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    tau = 13e-6
    Fc = 1e6
    Fs = 200e6
    Fd = 30e3
    
    if(doComplex):
        barkPulse, signal = barker(b, tau, Fs, Fc, amp = 1.0, dtype = "complex")
        dop_signal = barker(b, tau, Fs, 0, Fd, amp = 1.0, dtype = "complex")[1]
        matched_filt = np.conjugate(np.flip(signal))
    else:
        barkPulse, signal = barker(b, tau, Fs, Fc, dtype = "real")
        dop_signal = barker(b, tau, Fs, 0, Fd, dtype = "real")[1]
        matched_filt = np.flip(signal)
        
    output = np.convolve(signal, matched_filt)    
    matched_pulse = np.flip(barkPulse)
    pulse_output = np.convolve(barkPulse, matched_pulse)
    mismatched = np.convolve(dop_signal, matched_filt)
    t = np.linspace(0.0, 4 * tau, len(signal))
    t_m = np.linspace(0.0, 4 * tau, len(output))
    
    end_t = int(tau * Fs)
    ax = plt.subplot(3, 1, 1)
    ax.plot(t[0:end_t] / tau, np.real(signal[0:end_t]), 
            t[0:end_t] / tau, barkPulse[0:end_t], linewidth = 2)
    ax.plot(t[0:end_t] / tau, np.real(dop_signal[0:end_t]), linestyle = '--', color = 'k')
    ax.legend(["Recieved Signal", "Barker Code", "Doppler Shifted"], loc = "upper right")
    ax.set_xlim(0.0, t[end_t] / tau)
    ax.set_ylim(-1.25, 2.75)
    ax.grid(True)
    plt.xlabel("Normalized Time [t / tau]")
    plt.title("Barker Coded signal")
    # plt.ylabel("Barker Coded signal")
    
    
    ax = plt.subplot(3, 1, 2)
    ax.plot(t_m /tau, abs(output) / max(abs(pulse_output)), 
            t_m / tau, abs(pulse_output) / max(abs(pulse_output)), linewidth = 2)
    # ax.plot(t_m / tau, abs(mismatched) / max(abs(pulse_output)), linestyle = ':', color = 'k')
    # ax.legend(["Recieved Signal", "Barker Code", "Doppler Shifted"], loc = "upper right")
    ax.legend(["Recieved Signal", "Barker Code"], loc = "upper right")
    ax.set_xlim(1.25, 2.75)
    ax.grid(True)
    plt.xlabel("Normalized Time [t / tau]")
    plt.title("Matched Filter output")
    # plt.ylabel("Matched Filter output")
    
    ax = plt.subplot(3, 1, 3)
    ax.plot(t_m /tau, abs(mismatched) / max(abs(pulse_output)), 
            t_m / tau, abs(pulse_output) / max(abs(pulse_output)), linewidth = 2)
    # ax.plot(t_m / tau, abs(mismatched) / max(abs(pulse_output)), linestyle = ':', color = 'k')
    # ax.legend(["Recieved Signal", "Barker Code", "Doppler Shifted"], loc = "upper right")
    ax.legend(["Doppler Shifted", "Barker Code"], loc = "upper right")
    ax.set_xlim(1.25, 2.75)
    ax.grid(True)
    plt.xlabel("Normalized Time [t / tau]")
    plt.title("Matched Filter output")
    
    plt.subplots_adjust(hspace = 0.5)
    
    print("Range resolution of pulse with tau = {0:.6f} seconds equals = {1:.3f} meters".format(1e-6, range_resolution_pulse(1e-6)))
    print("Range resolution of modulated pulse with tau = {0:.6f} seconds, and chip width of {1:.6f} seconds,  equals = {2:.3f} meters".format(13e-6, 1e-6, range_resolution_pulse(1e-6)))
    
def problem_two(time_shift = True):
    tau = 1e-6 
    beta = 25e6
    Fs = 20 * beta 
    
    t = np.linspace(-tau / 2, tau / 2, int((tau / 2 - (-tau / 2)) * Fs)) 
    t_val = np.linspace(-tau / 2, tau / 2, len(t) * 4 - 1) 
    t_wind = np.linspace(0, tau, int((tau / 2 - (-tau / 2)) * Fs))
    t_plot = np.linspace(0, 2 * tau, int((tau / 2 - (-tau / 2)) * Fs) * 2)
    
    pulse_signal = np.zeros(int((tau / 2 - (-tau / 2)) * Fs) * 2)
    pulse_signal[0:int((tau / 2 - (-tau / 2)) * Fs)] = 1.0
    pulse_matched_filt = np.flip(pulse_signal)
    pulse_matched = np.convolve(pulse_signal, pulse_matched_filt)
    
    if(time_shift):
        mod_signal = np.exp(-1j * (pi * (beta / tau) * ((t - tau)**2) + pi / 4))
    else:
        mod_signal = np.exp(-1j * (pi * (beta / tau) * ((t)**2) + pi / 4)) 
    
    modSig_in = np.append(mod_signal, np.zeros(len(t), dtype = complex))
    modSig_conj = np.conjugate(np.flip(modSig_in)) 
    modSig_out = np.convolve(modSig_in, modSig_conj) 
    
    window_1 = np.sin(pi * (t_wind / tau))
    wind_sig_1 = np.zeros((len(t) * 2), dtype = complex)
    wind_sig_1[0:len(t)] = complex_multipy(mod_signal[0:len(t)], window_1)
    wind1_sig_filt = np.conjugate(np.flip(wind_sig_1))
    wind1_output = np.convolve(modSig_in, wind1_sig_filt)
    
    window_2 = np.sin(np.sin(pi * (t_wind / tau)))
    wind_sig_2 = np.zeros(len(t) * 2, dtype = complex)
    wind_sig_2[0:len(t)] = complex_multipy(mod_signal[0:len(t)], window_2)
    wind2_sig_filt = np.conjugate(np.flip(wind_sig_2))
    wind2_output = np.convolve(modSig_in, wind2_sig_filt)
    
    plt.figure()
    ax = plt.subplot(3, 1, 1)
    # ax.plot(t_plot, pulse_signal, linestyle = '--', color = 'k')
    ax.plot(t_plot, np.real(modSig_in), linewidth = 2)
    plt.xlim(0.0, t_plot[len(t_plot) - 1])
    ax.grid(True)
    # ax.legend(["Pulse signal", "LFM signal"], loc = "upper right")
    plt.xlabel("Normalized time [t / tau]")
    plt.ylabel("Real Amplitude")
    plt.title("LFM Waveform")
    
    ax = plt.subplot(3, 1, 2)
    # ax.plot(t_plot, pulse_matched_filt, linestyle = '--', color = 'k')
    ax.plot(t_plot, np.real(modSig_conj))
    ax.plot(t_plot, np.real(wind1_sig_filt), linewidth = 2)
    ax.plot(t_plot, np.real(wind2_sig_filt), linewidth = 2)
    plt.xlim(0.0, t_plot[len(t_plot) - 1])
    ax.grid(True)
    # ax.legend(["Matched Pulse", "Matched LFM", "Weights 1 Matched LFM", "Weights 2 Matched LFM"], loc = "upper left")
    ax.legend(["Matched LFM", "Weights 1 Matched LFM", "Weights 2 Matched LFM"], loc = "upper left")
    plt.xlabel("Normalized time [t / tau]")
    plt.ylabel("Real Amplitude")
    plt.title("Matched Filter")
    
    ax = plt.subplot(3, 1, 3)
    ax.plot(t_val / tau, abs(pulse_matched) / max(abs(pulse_matched)), linestyle = '--', color = 'k')
    ax.plot(t_val / tau, abs(modSig_out) / max(abs(modSig_out)), linewidth = 2) 
    ax.plot(t_val / tau, abs(wind1_output) / max(abs(wind1_output)))
    ax.plot(t_val / tau, abs(wind2_output) / max(abs(wind2_output)))
    #plt.xlim(t_val[0] / tau, t_val[len(t_val) - 1] / tau)
    ax.set_xlim(-0.3, 0.3)
    ax.grid(True)
    ax.legend(["Matched Pulse", "Matched LFM", "Weights 1 Matched LFM", "Weights 2 Matched LFM"], loc = "upper left")
    plt.xlabel("Normalized time [t / tau]")
    # plt.ylabel("Real Amplitude")
    plt.title("Matched Filter Output")
    
    plt.subplots_adjust(hspace = 0.5)
    
    plt.figure()
    f = np.linspace(-1.0, 1.0, len(np.fft.fft(modSig_in[0:len(modSig_in) >> 1])))
    modSig_fd = np.fft.fftshift(np.fft.fft(modSig_in[0:len(modSig_in) >> 1]))
    windSig_1_fd = np.fft.fftshift(np.fft.fft(wind_sig_1[0:len(wind_sig_1) >> 1]))
    windSig_2_fd = np.fft.fftshift(np.fft.fft(wind_sig_2[0:len(wind_sig_2) >> 1]))
    plt.plot(f, 10 * np.log10(abs(modSig_fd) / max(abs(modSig_fd))))
    plt.plot(f, 10 * np.log10(abs(windSig_1_fd) / max(abs(windSig_1_fd))))
    plt.plot(f, 10 * np.log10(abs(windSig_2_fd) / max(abs(windSig_2_fd))))
    plt.ylim(-15.0, 20.0)
    plt.xlim(-1.0, 1.0)
    plt.grid(True)
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude dB")
    plt.axhline(y = -3.0, color = "red", linewidth = 2, linestyle = "--")
    plt.legend(["LFM pulse", "LFM weight 1 pulse", "LFM weight 2 pulse", "-3dB points"], loc = "upper left")
    
    
    # plt.plot(abs(np.fft.fftshift(np.fft.fft(modSig_in))))
    # plt.plot(abs(np.fft.fftshift(np.fft.fft(wind_sig_1))))
    # plt.plot(abs(np.fft.fftshift(np.fft.fft(wind_sig_2))))

def problem_three():
    beta = 200e6
    tau = 1.5e-6
    # Fs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    # N = 1024
    
    # plt.figure()
    # for idx in range(0, len(Fs), 1):
    #     t = np.linspace(0.0, N / (Fs[idx] * beta), N)
    #     signal = np.sin(2 * pi * t * beta)
        
    #     ax = plt.subplot(len(Fs), 2, 2 * idx + 1)
    #     ax.plot(signal)
    #     ax = plt.subplot(len(Fs), 2, 2 * idx + 2)
    #     ax.plot(abs(np.fft.fft(signal)))
    #     ax.set_xlim(0, N / 2)
    #     plt.figure()
    #     plt.plot(abs(np.fft.fftshift(np.fft.rfft(signal))))
    #     plt.draw()
    #     plt.pause(0.01)    
    #     input("Press any key to continue...\n")
        
        
    
    print("Time bandwidth product: {0:.3f}".format(beta * tau))
    print("Range resolution: {0:.4f} meters".format(range_resolution_lfm(beta)))
    print("Minimum sample rate: {}".format(2 * beta))
    print("Number of samples per pulse: {}".format(2 * beta * tau))
    
    
if __name__ == "__main__":
    plt.close("all")
    problem_one()
    problem_two()
    problem_three()
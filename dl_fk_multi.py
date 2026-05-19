'''
This script performs Frequency-Wavenumber (F-K) migration for multistatic ultrasound data.

Acknowledgment & License Notice:
This file is a Python translation and modification of the MATLAB implementation `fk_multi.m` 
from the repository: https://gitlab.com/mj66/bentobox.

Copyright 2022, Marko Jakovljevic & Louise Zhuang
Licensed under the Apache License, Version 2.0. 

Changes made to the original work: 
- Translated the core signal processing pipeline from MATLAB to Python.
- Removed the original Stolt mapping interpolation block.
- Integrated a Deep Learning PyTorch inference step to replace the Stolt mapping.
'''

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import torch
import load_fk_model 


def _read_h5_scalar(x):
    '''Convert h5py scalar dataset/array to python float.'''
    x = np.array(x)
    return float(x.squeeze())


def iq_demod(sig, f_demod, fs):
    '''IQ demodulation'''
    T = sig.shape[0]
    t = np.arange(T, dtype=np.float64) / fs
    ref = np.exp(-2j * np.pi * f_demod * t)  # (T,)
    base = sig * ref[:, None]
    return np.real(base), np.imag(base)


def dl_fk_multi(signal, f_params, elempos, scale, start_time=0.0, speed_of_sound=1540.0,
                dl_batch_size=16):
    '''
    signal: (N_T, N_u, M) raw RF (real)
    f_params: dict with keys 'fs', 'f0'
    elempos: (N_u, 3) element positions
    start_time: scalar
    speed_of_sound: scalar

    Returns
    -------
    foc_data : (Nz, Nx) complex focused image
    t        : (Nz,) time axis
    x        : (Nx,) lateral axis
    '''

    fs = float(f_params["fs"])
    f0 = float(f_params["f0"])

    signal = np.asarray(signal)
    N_T, N_u, M = signal.shape

    du = np.mean(np.diff(elempos[:, 0]))

    # IQ Demodulation
    print("IQ Demodulation in process")
    sig2 = signal.reshape(N_T, -1)                 # (T, N_u*M)
    i_data, q_data = iq_demod(sig2, f0, fs)
    sig_cplx = (i_data + 1j * q_data).reshape(N_T, N_u, M)

    # Temporal shift / Causality
    print("Temporal Shift in process")
    ntshift = int(round(start_time * fs))
    t = np.arange(sig_cplx.shape[0], dtype=np.float64) / fs + start_time

    keep = t >= 0
    sig_cplx = sig_cplx[keep, :, :]
    t = t[keep]

    if ntshift < 0:
        ntshift = 0

    # 3D FFT (t,u,v) → (f, kv, ku)
    print("3D FFT in process")
    nT_FFT = max(4000, ntshift + N_T)
    if nT_FFT % 2 != 0:
        nT_FFT += 1

    # axial FFT over time
    signal_tmp1 = fftshift(fft(sig_cplx, n=nT_FFT, axis=0), axes=0)  # (nT_FFT, N_u, M)

    # time-frequency axis
    f_axes = (np.arange(-nT_FFT//2 + 1, nT_FFT//2 + 1, dtype=np.float64)
              * fs / nT_FFT)
    f_axes = f_axes + f0

    # account for offset in time domain
    freq_offset = np.exp(-2j * np.pi * f_axes * (start_time - np.mean(t)))
    signal_tmp1 *= freq_offset[:, None, None]

    # lateral zero-padding in u
    N_pad = int(round(N_u / 2))
    zero_pad_u = np.zeros((nT_FFT, N_pad, N_u), dtype=np.complex64)
    signal_tmp2 = np.concatenate([zero_pad_u, signal_tmp1, zero_pad_u], axis=1)
    nu_FFT = signal_tmp2.shape[1]

    # zero-padding in v
    zero_pad_v = np.zeros((nT_FFT, nu_FFT, N_pad), dtype=np.complex64)
    signal_tmp3 = np.concatenate([zero_pad_v, signal_tmp2, zero_pad_v], axis=2)

    # FFT over v (dim 1) → kv
    signal_tmp4 = fftshift(
        fft(fftshift(signal_tmp3, axes=1), n=nu_FFT, axis=1),
        axes=1
    )

    # FFT over u (dim 2) → ku
    signal_k = fftshift(
        fft(fftshift(signal_tmp4, axes=2), n=nu_FFT, axis=2),
        axes=2
    )

    nF, nKv, nKu = signal_k.shape

    # DL MAPPING ON ku SLICES (replaces Stolt)
    pred_b = np.zeros((nF, nKv, nKu), dtype=np.complex64)

    print("DL Mapping in process")
    for ii in range(nKu):
        print(f"[DL] Slice {ii+1} / {nKu}", flush=True)
        slice_ku = signal_k[:, :, ii]  # (nF, nKv) complex

        real_mat = np.ascontiguousarray(slice_ku.real.astype(np.float32))
        imag_mat = np.ascontiguousarray(slice_ku.imag.astype(np.float32))

        pred_re, pred_im = load_fk_model.run_inference(real_mat, imag_mat, scale)
        pred_b[:, :, ii] = pred_re + 1j * pred_im


    # Sum over ku
    signal_mapped = np.sum(pred_b, axis=2)  # (nF, nKv)

    # 2D IFFT: (kx,kz) → (x,z)
    print("2D IFFT in process")
    # axial ifft
    signal_ax_tmp = ifft(fftshift(signal_mapped, axes=0), axis=0)

    # axial shift
    time_shift = int(round(np.mean(t) * fs))
    signal_ax_tmp = np.roll(signal_ax_tmp, shift=time_shift, axis=0)
    signal_ax_tmp = signal_ax_tmp[:(N_T + ntshift), :]

    # lateral ifft
    foc_data = ifftshift(
        ifft(fftshift(signal_ax_tmp, axes=1), axis=1),
        axes=1
    )

    # keep only positive-time part + unpad laterally
    idc_positive = np.arange(1, N_T + 1) + ntshift
    idc_positive = idc_positive[idc_positive > 0]
    idc_positive = idc_positive[idc_positive <= foc_data.shape[0]]

    foc_data = foc_data[idc_positive - 1, N_pad:-N_pad]

    x = np.linspace(elempos[0, 0], elempos[-1, 0], foc_data.shape[1])

    return foc_data, t[:foc_data.shape[0]], x
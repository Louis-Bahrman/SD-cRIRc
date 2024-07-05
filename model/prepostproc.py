#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 04:31:39 2023

@author: louis
"""
import torchaudio
import torch
from torch import nn
from typing import Optional
import functools

import os
import sys

sys.path.append(os.getcwd() + "/model/base_models/FullSubNet/")

from model.base_models.FullSubNet.audio_zen.acoustics.mask import decompress_cIRM


class FullSubNetPreProc(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        waveform_length: int = 32767,
    ):
        super().__init__()
        self.waveform_length = waveform_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # self.spec_module = torchaudio.transforms.Spectrogram(
        # n_fft=n_fft, hop_length=hop_length, win_length=win_length, power=None  # , window_fn=torch.ones
        # )

    def get_stft(self, x):
        # return self.spec_module(x)
        return torch.stft(
            x[:, 0, ...], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True
        ).unsqueeze(1)

    def get_stft_features(self, x):
        x_complex = self.get_stft(x)
        return x_complex.abs(), x_complex.angle(), x_complex.real, x_complex.imag

    # We apply our own STFT, because if we did not, the window function would not match with the output.
    # In fact torch.istft(audio_zen_stft) gives the wrong result
    # self.stft = functools.partial(audio_zen_stft, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def forward(self, x):
        x_cropped = x[..., : self.waveform_length]  # if self.waveform_length else x
        mag, phase, real, imag = self.get_stft_features(x_cropped)
        return (mag, (real, imag)), None


class FullSubNetPostProc(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        waveform_len: int = 32767,
    ):
        super().__init__()
        self.istft = functools.partial(
            torch.istft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            length=waveform_len,
        )
        # self.istft = torchaudio.transforms.InverseSpectrogram(
        #     n_fft=n_fft, hop_length=hop_length, win_length=win_length  # , window_fn=torch.ones
        # )
        # self.waveform_len = waveform_len

    def invert_stft(self, stft):
        # stft B, C, F, T
        # return self.istft(stft, length=self.waveform_len)
        return self.istft(stft.squeeze(-3)).unsqueeze(-2)

    def get_stft(self, model_output, bypassed_features):
        cRM, (noisy_real, noisy_imag) = model_output
        cRM_decompressed = decompress_cIRM(cRM)
        # cRM_decompressed = cRM

        noisy_real = noisy_real[:, 0, ...]
        noisy_imag = noisy_imag[:, 0, ...]
        # enhanced_real = cRM[:, 0, None, ...] * noisy_real - cRM[:, 1, None, ...] * noisy_imag
        # enhanced_imag = cRM[:, 1, None, ...] * noisy_real + cRM[:, 0, None, ...] * noisy_imag
        # enhanced_stft = (enhanced_real + 1j * enhanced_imag)[..., 0, :, :]
        # cRM_complex = torch.view_as_complex(cRM.permute(0, 2, 3, 1).contiguous())
        # enhanced_stft = cRM_complex * torch.complex(noisy_real, noisy_imag)[:, 0, ...]
        enhanced_real = cRM_decompressed[..., 0] * noisy_real - cRM_decompressed[..., 1] * noisy_imag
        enhanced_imag = cRM_decompressed[..., 1] * noisy_real + cRM_decompressed[..., 0] * noisy_imag
        enhanced_stft = torch.complex(enhanced_real, enhanced_imag)
        return enhanced_stft.unsqueeze(-3)  # unsqueeze to match B, C, F, T shape

    def forward(self, model_output, bypassed_features):
        enhanced_stft = self.get_stft(model_output, bypassed_features)
        enhanced = self.invert_stft(enhanced_stft)
        # enhanced = self.istft(
        #     (enhanced_real, enhanced_imag),
        #     input_type="real_imag",
        # )
        # raise NotImplementedError("Pb de cputype complex ndims 5")
        # return 2.0 * enhanced
        return enhanced

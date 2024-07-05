#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:51:30 2023

@author: louis
"""

import os
import sys

sys.path.append(os.environ["HOME"] + "/dereverb-losses/model/base_models/FullSubNet/")
# sys.path.append(os.environ["HOME"] + "/dereverb-losses/model/base_models/FullSubNet_plus/speech_enhance")

from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask

import torch
from torch import nn


class FakeCIRM(nn.Module):
    """From the structure (magn, (real, imag)), computes the structure (cIRM, (real, imag)),
    such that FullSubnet(plus) postproc transforms (real, imag) in itself."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, t):
        if isinstance(t[0], tuple):
            # This is fullsubnet_plus
            x = t[0][0]
            cRM = build_complex_ideal_ratio_mask(
                torch.complex(torch.ones_like(x[:, 0, ...]), torch.ones_like(x[:, 0, ...])),
                torch.complex(torch.ones_like(x[:, 0, ...]), torch.ones_like(x[:, 0, ...])),
            )
            return cRM, t[1]

        else:
            x = t[0]
            cRM = build_complex_ideal_ratio_mask(
                torch.ones_like(x[:, 0, ...]),
                torch.ones_like(x[:, 0, ...]),
                torch.ones_like(x[:, 0, ...]),
                torch.ones_like(x[:, 0, ...]),
            )
            # cRM = compress_cIRM(torch.cat((torch.ones_like(x), torch.zeros_like(x)), dim=1))
            # return cRM.permute(0, 2, 3, 1), t[1]

            return cRM, t[1]

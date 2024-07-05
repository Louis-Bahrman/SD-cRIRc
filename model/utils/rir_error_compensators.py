#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:38:50 2024

@author: louis
"""

import torch
from torch import nn


class WienerFilter(nn.Module):
    def forward(self, mixture, noise, signal=None):
        if signal is None:
            signal = mixture - noise
        return mixture * signal.abs().square() / (noise.abs().square() + signal.abs().square())
        # return signal * signal.var(dim=(-1,-2)) / (noise.var(dim=(-1,-2)) + signal.var(dim=(-1,-2)))


class SpectralSubtraction(nn.Module):
    def __init__(self, power: bool = True):
        super().__init__()
        self.power = power

    def forward(self, mixture, noise, signal=None):
        if self.power:
            return (mixture.abs().square() - noise.abs().square()).abs().sqrt() * mixture.sgn()
        else:
            return (mixture.abs() - noise.abs()) * mixture.sgn()


class SimpleSubtraction(nn.Module):
    def forward(self, mixture, noise, signal=None):
        return mixture - noise

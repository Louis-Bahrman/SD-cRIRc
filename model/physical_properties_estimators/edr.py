#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:52:40 2024

@author: louis
"""

from model.physical_properties_estimators.edc import edc
from model.utils.tensor_ops import energy_to_db
from torch import nn


class EDR(nn.Module):
    TIME_FREQUENCY_INPUT=True
    def __init__(self, return_dB=True, scale="bandwise"):
        super().__init__()
        self.return_dB = return_dB
        self.scale = scale
        self.reestimate_from_oracle_rir = True

    def forward(self, rir_tf):
        edr = edc(rir_tf)
        if self.scale.lower() == "bandwise":
            edr = edr / edr[..., 0, None]
        elif self.scale.lower() == "global":
            edr = edr / edr[..., 0].mean(axis=-1, keepdim=True)[..., None]
        else:
            pass
        if self.return_dB:
            edr = energy_to_db(edr)
        return edr


# raise NotImplementedError("TODO implementer la loss sur les EDRs là où elles ne sont pas nulles")

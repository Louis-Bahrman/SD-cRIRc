#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 17:38:36 2023

@author: louis
"""

import sys
import os


# Import FullSubNet
sys.path.append(os.getcwd()+"/model/base_models/FullSubNet/")
from model.base_models.FullSubNet.recipes.dns_interspeech_2020.fullsubnet.model import Model as OriginalFullSubNet


class FullSubNet(OriginalFullSubNet):
    def forward(self, in_features):
        self.norm = lambda x: x
        if self.num_groups_in_drop_band != 1:
            raise NotImplementedError("Not tested yet")
        noisy_mag, bypassed_features = in_features
        cRM = super().forward(noisy_mag)
        cRM = cRM.permute(0, 2, 3, 1)
        return cRM, bypassed_features

    @property
    def last_layer(self):
        return self.sb_model.fc_output_layer.bias


if __name__ == "__main__":
    model = FullSubNet(
        num_freqs=257,
        # look_ahead=2,
        look_ahead=5,
        sequence_model="LSTM",
        fb_num_neighbors=0,
        sb_num_neighbors=15,
        fb_output_activate_function="ReLU",
        sb_output_activate_function=False,
        fb_model_hidden_size=512,
        sb_model_hidden_size=384,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=1,
        # num_groups_in_drop_band=2,
        weight_init=False,
    )

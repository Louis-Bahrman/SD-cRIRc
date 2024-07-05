#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:07:58 2023

@author: louis
"""

from torch import nn


class Identity(nn.Identity):
    def __init__(self):
        super().__init__()
        self.reestimate_from_oracle_rir = True

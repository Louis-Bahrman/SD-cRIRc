#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:35:29 2023

@author: louis
"""

import torch
from torch import nn
from model.physical_properties_estimators.identity import Identity
from model.utils.tensor_ops import energy_to_db


def edc(rir):
    power = rir.abs() ** 2
    return torch.flip(torch.cumsum(torch.flip(power, (-1,)), -1), (-1,))


def edc_db(rir, epsilon=0):
    energy = edc(rir)
    if epsilon <= 0:
        assert rir.squeeze().ndim == 1, rir.shape
        energy = energy[energy > 0]
    else:
        energy += epsilon
    edc_db = energy_to_db(energy)
    return edc_db


def edc_db_scaled(rir, return_edc_sum=False, epsilon=0):
    energy_db = edc_db(rir, epsilon=epsilon)
    energy_scaled = energy_db - energy_db[..., 0, None]
    if return_edc_sum:
        return energy_scaled, energy_db[..., 0]
    return energy_scaled


class EDC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rir):
        return edc(rir)


class EDCdB(nn.Module):
    def __init__(self, epsilon: float = 1e-12):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, rir):
        return edc_db_scaled(rir, epsilon=self.epsilon)


if __name__ == "__main__":
    from datasets import SynthethicRirDataset, LibrispeechSimulatedRirDataModule
    from model.physical_properties_estimators.early_echoes import EarlyEchoes

    rir_dataset = SynthethicRirDataset()
    dataModule = LibrispeechSimulatedRirDataModule(
        batch_size=8,
        rir_dataset=rir_dataset,
        dry_signal_start_index_train=None,
    )
    dataModule.prepare_data()
    dataModule.setup()
    loader = dataModule.train_dataloader()
    y, (x, rir_properties, rir) = next(iter(loader))
    edc_db_module = EDCdB()
    edc = edc_db_module(rir)
    edc_scaled = edc - edc[..., 0, None]

    import matplotlib.pyplot as plt

    plt.close("all")
    plt.figure()
    plt.title("EDC")
    plt.plot(edc.numpy().squeeze().T)
    plt.show()

    plt.figure()
    plt.title("EDC scaled")
    plt.plot(edc_scaled.numpy().squeeze().T)
    plt.show()

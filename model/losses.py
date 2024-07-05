#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:17:50 2024

@author: louis
"""
import torch
import torch.nn
import torch.nn as nn
from model.base_models.FullSubNet.audio_zen.acoustics.mask import build_complex_ideal_ratio_mask


class FilteredLoss(nn.Module):
    """Compares preds and target only where min < target < max
    Strict inequality to also avoid inf and nans
    can create a mask only on target or on preds too
    """

    def __init__(
        self,
        min: float = -torch.inf,
        max: float = torch.inf,
        base_loss: nn.Module = nn.MSELoss(),
        consider_only_target_mask: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.min = min
        self.max = max
        self.filtering_op = lambda t: torch.logical_and(t > self.min, t < self.max)
        self.consider_only_target_mask = consider_only_target_mask

    def forward(self, pred, target):
        assert pred.shape == target.shape, (pred.shape, target.shape)
        mask_target = self.filtering_op(target)
        mask_pred = self.filtering_op(pred)
        if self.consider_only_target_mask:
            mask_to_consider = mask_target
        else:
            mask_to_consider = torch.logical_and(mask_pred, mask_target)
        masked_preds = pred[mask_to_consider]
        masked_target = target[mask_to_consider]
        mask_finite = torch.logical_and(masked_preds.isfinite(), masked_target.isfinite())
        return self.base_loss(masked_preds[mask_finite], masked_target[mask_finite])


class ComplexToRealMSELoss(nn.Module):
    def forward(self, y, x):
        return torch.dist(y, x) / x.numel()


class ZeroLoss(nn.Module):
    # Default loss always returns zero
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return 0.0


class FullSubNetLoss(nn.Module):
    def __init__(
        self,
        num_groups_in_drop_band: int = 1,
        weight_real: float = 1.0,
        weight_imag: float = 1.0,
        weight_phase: float = 0.0,
    ):
        super().__init__()
        self.loss_function = nn.MSELoss()
        self.weight_real = weight_real
        self.weight_imag = weight_imag
        self.weight_phase = weight_phase
        # assert num_groups_in_drop_band == 1
        # self.num_groups_in_drop_band = num_groups_in_drop_band

    def forward(self, preds, targets):
        # from fullsubnet trainer
        cRM, (noisy_real, noisy_imag) = preds
        _, (clean_real, clean_imag) = targets
        cIRM = build_complex_ideal_ratio_mask(
            noisy_real=noisy_real[:, 0, ...],
            noisy_imag=noisy_imag[:, 0, ...],
            clean_real=clean_real[:, 0, ...],
            clean_imag=clean_imag[:, 0, ...],
        )  # [B, F, T, 2]
        # cRM = cRM.permute(0, 2, 3, 1)
        loss = 0
        if self.weight_real != 1.0 or self.weight_imag != 1.0:
            loss_real = self.loss_function(cRM[..., 0], cIRM[..., 0])
            loss_imag = self.loss_function(cRM[..., 1], cIRM[..., 1])
            loss += self.weight_real * loss_real + self.weight_imag * loss_imag
        else:
            loss += self.loss_function(cIRM, cRM)
        if self.weight_phase > 0.0:
            loss += self.loss_function(
                torch.view_as_complex(cIRM).angle(), torch.view_as_complex(cRM.contiguous()).angle()
            )
        return loss

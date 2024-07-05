#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:39:01 2023

@author: louis
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
    SpeechReverberationModulationEnergyRatio,
    ScaleInvariantSignalDistortionRatio,
)
from lightning.pytorch.utilities import grad_norm

from model.grad_balancers import AbsGradBalancer
from model.utils.fake_cirm import FakeCIRM


class FullModel(L.LightningModule):
    def __init__(
        self,
        preproc: nn.Module,
        base_model: nn.Module,
        postproc: nn.Module,
        nonblind_rir_estimator: nn.Module,
        physical_properties_estimator: nn.Module,
        base_model_loss: nn.Module,
        signal_loss: nn.Module,
        physical_properties_loss: nn.Module,
        base_model_loss_weight_function: str,
        signal_loss_weight_function: str,
        physical_properties_loss_weight_function: str,
        fs: int = int(16e3),
        recompute_k: bool = False,  # compute k from y and x
        metrics: list[str] = ["stoi", "stoi_input"],  # ["stoi", "pesq", "srmr"]
        rir_estimation_error_compensator: nn.Module | None = None,
        grad_balancer: AbsGradBalancer | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            # "base_model_loss_weight_function",
            # "signal_loss_weight_function",
            # "physical_properties_loss_weight_function", # they are ignored in the logger in any case
            ignore=[
                "preproc",
                "base_model",
                "postproc",
                "nonblind_rir_estimator",
                "physical_properties_estimator",
                "base_model_loss",
                "signal_loss",
                "physical_properties_loss",
                "rir_estimation_error_compensator",
                "grad_balancer",
            ],
            logger=True,
        )

        # From https://github.com/Lightning-AI/lightning/issues/11494
        self.save_hyperparameters(ignore=[], logger=False)

        self.preproc = preproc
        self.base_model = base_model
        self.postproc = postproc
        self.nonblind_rir_estimator = nonblind_rir_estimator
        self.physical_properties_estimator = physical_properties_estimator

        if self.hparams.recompute_k:
            self.rir_estimation_error_compensator = rir_estimation_error_compensator

        # Losses
        self.base_model_loss = base_model_loss
        self.signal_loss = signal_loss
        self.physical_properties_loss = physical_properties_loss
        self.grad_balancer = grad_balancer
        if self.grad_balancer is not None and not self.grad_balancer.apply_at_backward:
            self.grad_balancer.last_layer = self.base_model.last_layer

        self.current_base_model_loss_weight = eval(self.hparams.base_model_loss_weight_function)
        self.current_signal_loss_weight = eval(self.hparams.signal_loss_weight_function)
        self.current_physical_properties_loss_weight = eval(self.hparams.physical_properties_loss_weight_function)

        # for logging
        self.stoi = ShortTimeObjectiveIntelligibility(fs=self.hparams.fs)  # , extended=True)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=self.hparams.fs, mode="wb")
        self.srmr = SpeechReverberationModulationEnergyRatio(fs=self.hparams.fs, fast=True)
        self.sisdr = ScaleInvariantSignalDistortionRatio()

        # Other properties
        self.hparams.estimate_rir_tf = getattr(self.nonblind_rir_estimator, "TIME_FREQUENCY_INPUT", False)
        self.base_model.is_fullsubnet = "FullSubNet" in type(self.base_model).__name__
        if self.base_model.is_fullsubnet:
            self.fake_cIRM = FakeCIRM()
        self.reestimate_properties_from_oracle_rir = getattr(
            self.physical_properties_estimator, "reestimate_from_oracle_rir", True
        )
        # self.example_input_array = torch.normal(
        #     0, 1, (8, 1, self.postproc.inverse_spec_module_angle_module.waveform_length)
        # )
        # self.example_input_array = torch.sin(
        #     torch.arange(1, 2).outer(torch.arange(2 * self.preproc.waveform_len)).unsqueeze(1)
        # )

    def dereverberate_only(self, y):
        Y_features, bypassed_features = self.preproc(y)
        X_hat_features = self.base_model(Y_features)
        x_hat = self.postproc(X_hat_features, bypassed_features)
        return x_hat

    def forward(self, y):
        Y_features, bypassed_features = self.preproc(y)
        X_hat_features = self.base_model(Y_features)
        x_hat = self.postproc(X_hat_features, bypassed_features)
        k_hat = self._compute_k(y, x_hat, X_hat_features, bypassed_features, k_is_exact=False)
        estimated_physical_properties = self.physical_properties_estimator(k_hat)
        return (x_hat, estimated_physical_properties, k_hat)

    def compute_and_log_full_loss(
        self,
        x_hat,
        X_hat_features,
        estimated_physical_properties,
        x,
        X_features,
        physical_properties,
        log=True,
        mode_prefix="train",
    ):
        current_model_loss = self.base_model_loss(X_hat_features, X_features)
        current_signal_loss = self.signal_loss(x_hat, x)
        current_physical_properties_loss = self._compute_physical_properties_loss_if_necessary(
            estimated_physical_properties, physical_properties
        )
        signal_and_model_loss = (
            self.current_base_model_loss_weight * current_model_loss
            + self.current_signal_loss_weight * current_signal_loss
        )

        balanced_signal_and_model_loss, balanced_physical_properties_loss = self._apply_grad_balancer_if_necessary(
            signal_and_model_loss, current_physical_properties_loss
        )

        full_loss = (
            balanced_signal_and_model_loss
            + self.current_physical_properties_loss_weight * balanced_physical_properties_loss
        )
        if log:
            dict_to_log = {
                mode_prefix + "_base_model_loss": current_model_loss,
                mode_prefix + "_signal_loss": current_signal_loss,
                mode_prefix + "_physical_properties_loss": current_physical_properties_loss,
                mode_prefix + "_signal_and_model_loss": signal_and_model_loss,
                mode_prefix + "_signal_and_model_grad_weights": balanced_signal_and_model_loss / signal_and_model_loss,
                mode_prefix
                + "_physical_properties_grad_weights": balanced_physical_properties_loss
                / current_physical_properties_loss,
                mode_prefix + "_balanced_signal_and_model_loss": balanced_signal_and_model_loss,
                mode_prefix + "_balanced_physical_propreties_loss": balanced_physical_properties_loss,
                mode_prefix + "_full_loss": full_loss,
            }
            self.log_dict(dict_to_log, on_epoch=True)
        if self.training and self.current_physical_properties_loss_weight != 0 and not full_loss.isfinite():
            return None
        return full_loss

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        if not getattr(self.grad_balancer, "apply_at_backward", False):
            super().optimizer_zero_grad(epoch, batch_idx, optimizer)

    def backward(self, loss, *args, **kwargs):
        if not getattr(self.grad_balancer, "apply_at_backward", False):
            super().backward(loss, *args, **kwargs)

    def _apply_grad_balancer_if_necessary(self, signal_and_model_loss, current_physical_properties_loss):
        if self.grad_balancer is not None and self.current_physical_properties_loss_weight != 0:
            (balanced_signal_and_model_loss, balanced_physical_properties_loss), new_grads = self.grad_balancer(
                (signal_and_model_loss, current_physical_properties_loss), model_parameters=self.base_model.parameters()
            )
            if new_grads is not None:
                # Update model gradients. In the other case it means that the
                for p, ng in zip(self.base_model.parameters(), new_grads):
                    p.grad = ng
        else:
            balanced_signal_and_model_loss = signal_and_model_loss
            balanced_physical_properties_loss = current_physical_properties_loss
        return balanced_signal_and_model_loss, balanced_physical_properties_loss

    def _compute_x_hat_if_necessary(self, X_hat_features, bypassed_features, x):
        # x only used to be copied if not necessary
        if self.training and self.hparams.estimate_rir_tf and self.current_signal_loss_weight == 0:
            return torch.ones_like(x)  # less risk of dividing by 0
        else:
            return self.postproc(X_hat_features, bypassed_features)

    def _compute_physical_properties_loss_if_necessary(self, estimated_physical_properties, physical_properties):
        if self.training and self.current_physical_properties_loss_weight == 0:
            # both estimated_physical_properties and physical_properties should be oracle_physical_properties
            return torch.dist(estimated_physical_properties, physical_properties)
        return self.physical_properties_loss(estimated_physical_properties, physical_properties)

    def training_step(self, batch, batch_idx):
        y, (x, physical_properties, k) = batch
        # assert not y.isnan().any(), "check convolve_on_gpu"
        # almost same as forward but we compute the loss on x_hat_features also
        Y_features, Y_bypassed_features = self.preproc(y)
        X_features, X_bypassed_features = self.preproc(x)

        bypassed_features = self._merge_bypassed_features(X_bypassed_features, Y_bypassed_features)
        X_hat_features = self.base_model(Y_features)
        x_hat = self._compute_x_hat_if_necessary(X_hat_features, bypassed_features, x)
        k, k_recomputed, k_hat = self._compute_k_and_k_hat(
            y, x, x_hat, Y_features, X_features, X_hat_features, bypassed_features, k
        )
        # Add step of modeling error compensation
        k_recomputed_compensated, k_hat_compensated = self._compensate_rir_estimation_error(k, k_recomputed, k_hat)

        target_physical_properties, pred_physical_properties = self._compute_pred_and_target_physical_properties(
            k_hat_compensated, k_recomputed_compensated, k, physical_properties
        )

        full_loss = self.compute_and_log_full_loss(
            x_hat,
            X_hat_features,
            pred_physical_properties,
            x,
            X_features,
            target_physical_properties,
            log=True,
            mode_prefix="train",
        )
        return full_loss

    def _compute_metric(self, metric_name, x_hat, y, x):
        metric_arguments = {"preds": x_hat}
        if "input" in metric_name:  # We compute the metric of the wet signal
            metric_arguments["preds"] = y[..., : x.shape[-1]]
        if "srmr" not in metric_name:
            # we add target
            metric_arguments["target"] = x
        metric_attr = metric_name.split("_")[0]
        return getattr(self, metric_attr)(**metric_arguments)

    def validation_step(self, val_batch, batch_idx):
        y, (x, physical_properties, k) = val_batch
        # almost same as forward but we compute the loss on x_hat_features also
        Y_features, bypassed_features = self.preproc(y)
        X_hat_features = self.base_model(Y_features)
        x_hat = self.postproc(X_hat_features, bypassed_features)
        X_features, _ = self.preproc(x)
        k, k_recomputed, k_hat = self._compute_k_and_k_hat(
            y, x, x_hat, Y_features, X_features, X_hat_features, bypassed_features, k
        )
        k_recomputed_compensated, k_hat_compensated = self._compensate_rir_estimation_error(k, k_recomputed, k_hat)

        target_physical_properties, pred_physical_properties = self._compute_pred_and_target_physical_properties(
            k_hat_compensated, k_recomputed_compensated, k, physical_properties
        )
        full_loss = self.compute_and_log_full_loss(
            x_hat,
            X_hat_features,
            pred_physical_properties,
            x,
            X_features,
            target_physical_properties,
            log=True,
            mode_prefix="val",
        )

        val_only_metrics_dict = {
            "current_physical_properties_loss_weight": self.current_physical_properties_loss_weight,
        }
        val_only_metrics_dict.update(
            {
                "val_" + metric_name: self._compute_metric(metric_name, x_hat, y, x)
                for metric_name in self.hparams.metrics
            }
        )

        self.log_dict(val_only_metrics_dict, on_epoch=True)

        if batch_idx == 0:
            self._log_audios(
                {
                    "x": x,
                    "x_hat": x_hat,
                    "y": y,
                    "k": k,
                    "k_hat": k_hat,
                    "k_recomputed": k_recomputed,
                },
                scale=True,
            )

            hparams_dict = {
                "model_loss_weight_function": self.hparams.base_model_loss_weight_function,
                "signal_loss_weight_function": self.hparams.signal_loss_weight_function,
                "physical_properties_loss_weight_function": self.hparams.physical_properties_loss_weight_function,
                "current_physical_properties_loss_weight": self.current_physical_properties_loss_weight,
                "current_signal_loss_weight": self.current_signal_loss_weight,
                "current_base_model_loss_weight": self.current_base_model_loss_weight,
                "use_oracle_phase_training": getattr(self.postproc, "use_oracle_phase_during_training", False),
            }
            self.logger.log_hyperparams(
                hparams_dict,
                metrics={"hp/stoi": val_only_metrics_dict.get("val_stoi", 0.0)},
            )
        return full_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=0.0001, betas=(0.9, 0.999))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": OneCycleLR(optimizer, max_lr=0.001, total_steps=self.trainer.estimated_stepping_batches)
                # "scheduler": ReduceLROnPlateau(optimizer, mode="min", patience=1, verbose=True),
                # "monitor": "val_full_loss",
            },
        }
        # return optimizer

    def _log_audios(self, waveform_batches_dict, scale=False):
        tensorboard = self.logger.experiment
        for k, v in waveform_batches_dict.items():
            if self.hparams.estimate_rir_tf and v.ndim > 3:
                # v is a stft so we invert it
                v = self.postproc.invert_stft(v)
            wav = v[0, 0, ...]
            if scale:
                wav = wav / abs(wav).max()
            tensorboard.add_audio(k, wav, global_step=self.global_step, sample_rate=self.hparams.fs)
        tensorboard.close()

    def on_validation_epoch_end(self):
        # Update weights
        self.current_base_model_loss_weight = eval(self.hparams.base_model_loss_weight_function)
        self.current_signal_loss_weight = eval(self.hparams.signal_loss_weight_function)
        self.current_physical_properties_loss_weight = eval(self.hparams.physical_properties_loss_weight_function)

    def _recompute_k_exact_fullsubnet(self, y, X_features):
        assert "FullSubNet" in type(self.base_model).__name__
        perfect_X_hat_features = self.fake_cIRM(X_features)
        # attention les bypassed featuress corespondent à celles de Y, donc il faut prendre celles de X.
        # Heureusement qu'on peut quand meme s'en sortir car la magnitude et phase de X sont transmises.
        # Il faut appliquer postproc avec None
        perfect_x_hat = self.postproc(perfect_X_hat_features, None)
        perfect_k_hat = self.nonblind_rir_estimator(y, perfect_x_hat)
        return perfect_k_hat

    def _compute_K_stft(self, y, X_features, bypassed_features, k_is_exact: bool):
        Y_stft = self.preproc.get_stft(y)
        if self.base_model.is_fullsubnet and k_is_exact:
            X_features = self.fake_cIRM(X_features)
        X_stft = self.postproc.get_stft(X_features, bypassed_features)
        K_stft = self.nonblind_rir_estimator(Y_stft, X_stft)
        return K_stft

    def _compute_k(self, y, x, X_features, bypassed_features, k_is_exact: bool):
        # Also valid for k_hat
        # TF
        if self.hparams.estimate_rir_tf:
            return self._compute_K_stft(y, X_features, bypassed_features, k_is_exact=k_is_exact)
        # else (time domain) if fullsubnet, x is exact (meaning we need to use the fake_cirm)
        # In fact we could just apply FakeCIRM if k_is_exact in any case (wether it is tf inversion or not)
        if self.base_model.is_fullsubnet and k_is_exact:
            return self._recompute_k_exact_fullsubnet(y, X_features)
        # else: Time domain
        return self.nonblind_rir_estimator(y, x)

    def _compute_k_and_k_hat(self, y, x, x_hat, Y_features, X_features, X_hat_features, bypassed_features, k):
        if self.training and self.current_physical_properties_loss_weight == 0:
            return k, k, k
        # Exception for fullsubnet for which
        if self.hparams.estimate_rir_tf:
            k = self.preproc.get_stft(k)
        if self.hparams.recompute_k:
            k_recomputed = self._compute_k(y, x, X_features, bypassed_features, k_is_exact=True)
        else:
            k_recomputed = k
        k_hat = self._compute_k(y, x_hat, X_hat_features, bypassed_features, k_is_exact=False)
        return k, k_recomputed, k_hat

    def _merge_bypassed_features(self, X_bypassed_features, Y_bypassed_features):
        if getattr(self.postproc, "use_oracle_phase_during_training", False):
            if getattr(self.preproc, "normalize", False):
                bypassed_features = (X_bypassed_features[0], Y_bypassed_features[1])
            else:
                bypassed_features = X_bypassed_features
        else:
            bypassed_features = Y_bypassed_features
        return bypassed_features

    def _compute_target_physical_properties(self, k_recomputed_compensated, k, oracle_physical_properties):
        # physical_properties transmitted with batch
        if self.hparams.recompute_k:
            return self.physical_properties_estimator(k_recomputed_compensated)
        else:
            # We use k transmitted by dataloader
            if self.reestimate_properties_from_oracle_rir:
                return self.physical_properties_estimator(k)
            else:
                return oracle_physical_properties

    def _compute_pred_and_target_physical_properties(
        self, k_hat_compensated, k_recomputed_compensated, k, oracle_physical_properties
    ):
        if self.training and self.current_physical_properties_loss_weight == 0:
            return oracle_physical_properties, oracle_physical_properties
        pred_physical_properties = self.physical_properties_estimator(k_hat_compensated)
        target_physical_properties = self._compute_target_physical_properties(
            k_recomputed_compensated, k, oracle_physical_properties
        )
        return target_physical_properties, pred_physical_properties

    def _compensate_rir_estimation_error(self, k, k_recomputed, k_hat):
        if self.hparams.recompute_k and self.rir_estimation_error_compensator is not None:
            k_recomputed_compensated = self.rir_estimation_error_compensator(
                mixture=k_recomputed, noise=k_recomputed - k, signal=k
            )
            k_hat_compensated = self.rir_estimation_error_compensator(mixture=k_hat, noise=k_recomputed - k)
            return k_recomputed_compensated, k_hat_compensated
        else:
            return k_recomputed, k_hat

    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.base_model, norm_type=2)
    #     self.log_dict(norms, on_epoch=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:47:13 2024

@author: louis
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:48:20 2024

@author: louis
"""

from model.utils.tensor_ops import toeplitz, fftconvolve, zero_pad, nansum_complex, solve_lstsq_qr
import torch
from torch import nn

class NonBlindRIREstimator(nn.Module):
    def __init__(self, epsilon: float = 1e-10):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y, x_hat):
        n_y = y.shape[-1]
        n_x = x_hat.shape[-1]
        n_k = n_y - n_x + 1
        Y = torch.fft.rfft(y, n_y)
        X_hat = torch.fft.rfft(x_hat, n_y)
        return torch.fft.irfft(Y / (X_hat + self.epsilon), n_y)[..., :n_k]

class CTF(nn.Module):
    def __init__(
        self,
        epsilon_lambda: float = 1.0,
        output_len: int = 64,
        num_absolute_cross_bands: int = 0,
        pad_X_hat_before_unfold: bool = True,
        crop_toeplitz: bool = False,
        return_stft: bool = False,
        solve_qr: bool = True,  # Use version that is faster at backward
    ):
        super().__init__()
        self.TIME_FREQUENCY_INPUT = True

        self.epsilon_lambda = epsilon_lambda
        self.output_len = output_len
        self.num_absolute_cross_bands = num_absolute_cross_bands
        self.pad_X_hat_before_unfold = pad_X_hat_before_unfold
        self.crop_toeplitz = crop_toeplitz
        self.return_stft = return_stft
        self.solve_qr = solve_qr
        self.solve_if_almost_determined = False
        self.num_total_bands = 2 * num_absolute_cross_bands + 1
        if not self.pad_X_hat_before_unfold:
            raise NotImplementedError("todo reconstruction and shape")

    def compute_lambda(self, Y_tf):
        return torch.maximum(
            self.epsilon_lambda * Y_tf.abs().square().amax(dim=(-1, -2), keepdim=True),
            Y_tf.abs().square(),
        )

    def compute_X_hat_padded(self, X_hat):
        if self.pad_X_hat_before_unfold and self.num_absolute_cross_bands > 0:
            F_prim = self.num_absolute_cross_bands
            return torch.cat(
                (
                    # X_hat[..., 1 : F_prim + 1, :].flip(-2).conj(),
                    1e-8 * torch.ones_like(X_hat[..., :F_prim, :]),
                    X_hat,
                    # X_hat[..., -F_prim:, :].flip(-2).conj(),  # Use the cyclic property of FCP
                    # torch.zeros_like(X_hat[..., :F_prim, :]),
                    1e-8 * torch.ones_like(X_hat[..., :F_prim, :]),
                ),
                dim=-2,
            )  # F + 2F' For zero-padding and hermitian symmetry
        return X_hat

    def compute_X_hat_unfolded(self, X_hat):
        X_hat_padded = self.compute_X_hat_padded(X_hat)
        return X_hat_padded.unfold(-2, self.num_total_bands, 1)  # F, T, F'
        #        X_hat_padded=X_hat

    #        X_hat_padded=torch.cat((
    #          X_hat[..., 1:F_prim+1,:].flip(-2).conj(),
    #            X_hat,
    #            X_hat[..., 0:F_prim+0,:].conj(),
    #        ), dim=-2) # F + 2F' For zero-padding and hermitian symmetry

    def compute_modified_toeplitz(self, Y, X_hat):
        assert self.num_total_bands * self.output_len <= X_hat.size(-1), "under-determined"
        F, N_X, N_Y, N_K, F_prim = (
            X_hat.size(-2),
            X_hat.size(-1),
            Y.size(-1),
            self.output_len,
            self.num_absolute_cross_bands,
        )
        X_hat_unfolded = self.compute_X_hat_unfolded(X_hat)
        return torch.cat(
            tuple(toeplitz(zero_pad(X_hat_unfolded[..., i], N_Y))[..., :N_K] for i in range(X_hat_unfolded.size(-1))),
            dim=-1,
        )  # F, N_y, N_K*F'

    def crop__modified_toeplitz(self, X_toeplitz, Y):
        return (
            X_toeplitz[..., self.output_len - 1 : -self.output_len + 1, :],
            Y[..., self.output_len - 1 : -self.output_len + 1],
        )

    def center_Y(self, Y):
        if self.num_absolute_cross_bands > 0:
            return Y[..., : -2 * self.num_absolute_cross_bands, :]
        return Y

    def forward(self, Y_tf, X_hat_tf, lambda_tf=None):
        if lambda_tf is None:
            lambda_tf = self.compute_lambda(Y_tf)
        Y = Y_tf / torch.sqrt(lambda_tf)[..., : Y_tf.size(-2), : Y_tf.size(-1)]
        X_hat = X_hat_tf / torch.sqrt(lambda_tf)[..., : X_hat_tf.size(-2), : X_hat_tf.size(-1)]
        # if self.num_absolute_cross_bands==0:
        #     # classical FCP, so we can afford to use deconvolve_corr
        #     return deconvolve_corr(Y, X_hat, output_len=self.output_len).unsqueeze(-1)
        X_hat_toeplitz = self.compute_modified_toeplitz(Y, X_hat)
        if self.crop_toeplitz:
            X_hat_toeplitz, Y = self.crop__modified_toeplitz(X_hat_toeplitz, Y)
        if not self.pad_X_hat_before_unfold:
            Y = self.center_Y(Y)
        try:
            if self.solve_if_almost_determined and X_hat_toeplitz.size(-1) == Y.size(-1) - self.output_len:
                # The system is almost exactly determmined so we use solve
                X_hat_toeplitz_reshaped = X_hat_toeplitz[..., : X_hat_toeplitz.size(-1), :]
                diagonal = 1e-8 * torch.eye(
                    X_hat_toeplitz_reshaped.size(-1),
                    device=X_hat_toeplitz_reshaped.device,
                    dtype=X_hat_toeplitz_reshaped.dtype,
                )
                X_hat_toeplitz_to_solve = X_hat_toeplitz_reshaped + diagonal
                sol = torch.linalg.solve(X_hat_toeplitz_to_solve, Y[..., : X_hat_toeplitz.size(-1)])
            elif self.solve_qr:
                sol = solve_lstsq_qr(X_hat_toeplitz, Y)
            else:
                lstsq_res = torch.linalg.lstsq(X_hat_toeplitz, Y)
                sol = lstsq_res.solution
        except:
            sol = torch.full_like(X_hat_toeplitz[..., 0, :], torch.nan)
        sol_reshaped = torch.stack(sol.chunk(self.num_absolute_cross_bands * 2 + 1, dim=-1), dim=-1)
        # sol_reshaped[torch.logical_not(sol_reshaped.isfinite())] = 0
        if self.return_stft:
            return self.reconstruct_stft(sol_reshaped)
        return sol_reshaped

    def apply_on_dry(self, X_hat_tf, fcp_result):
        X_hat_unfolded = self.compute_X_hat_unfolded(X_hat_tf)
        return nansum_complex(fftconvolve(X_hat_unfolded, fcp_result, dim=-2), dim=-1)

    def plot_fcp(self, fcp_result, vmin=None, vmax=None):
        import matplotlib.pyplot as plt
        from debug.plot_spectrogram import make_plottable

        num_total_bands = fcp_result.shape[-1]
        assert num_total_bands == self.num_absolute_cross_bands * 2 + 1
        fig, axs = plt.subplots(
            nrows=2, ncols=num_total_bands, sharex=True, sharey=True, squeeze=False, figsize=(10, 8)
        )
        fig.tight_layout()
        for iax, (ax_magn, ax_phase) in enumerate(axs.T):
            im_magn = ax_magn.imshow(
                make_plottable(fcp_result[..., iax], keep_sign=False, squeeze=(0, 1)),
                vmin=vmin,
                vmax=vmax,
                origin="lower",
            )
            fig.colorbar(im_magn, ax=ax_magn)
            im_phase = ax_phase.imshow(
                make_plottable(fcp_result[..., iax].angle(), db_scale=False, squeeze=(0, 1)), origin="lower", cmap="hsv"
            )
            fig.colorbar(im_phase, ax=ax_phase)

    def reconstruct_stft(self, fcp_result):
        # /2 for the middle of the STFT window
        multiplicator = torch.exp(1j * torch.pi * torch.arange(fcp_result.size(-3), device=fcp_result.device)).reshape(
            1, 1, -1, 1
        )  # F
        # multiplicator = torch.pow(-1.0, torch.arange(fcp_result.size(-3)))
        multiplicator_unfolded = self.compute_X_hat_unfolded(multiplicator)  # B, C, F, T -> unfolded: B, C, F, T, F'
        return (fcp_result * multiplicator_unfolded).sum(axis=-1)

    def filter_prediction(self, Y, X_hat, fcp_filter=None, lambda_tf=None):
        if fcp_filter is None:
            previous_value_return_stft = self.return_stft
            self.return_stft = False
            fcp_filter = self.forward(Y, X_hat, lambda_tf=lambda_tf)
            self.return_stft = previous_value_return_stft
        applied_filter = self.apply_on_dry(X_hat, fcp_filter)
        return Y[..., : X_hat.size(-1)] - (applied_filter[..., : X_hat.size(-1)] - X_hat)

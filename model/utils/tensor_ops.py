#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:04:45 2024

@author: louis
"""

import torch
import math
from torchaudio.functional import fftconvolve as fftconvolve_torchaudio
from torchaudio.functional.functional import _check_shape_compatible, _check_convolve_mode, _apply_convolve_mode


def power_to_db(x):
    #     if x == 0:
    #         return -torch.inf
    return 10 * torch.log10(x)


energy_to_db = power_to_db


def db_to_power(x_db):
    return torch.pow(10, x_db / 10)


def signal_power_db(signal):
    return power_to_db((torch.abs(signal) ** 2).mean())


def complex_circular_gaussian_noise(mean=0 + 0j, std=math.sqrt(2), **kwargs):
    return torch.complex(torch.normal(mean.real, std, **kwargs), torch.normal(mean.imag, std, **kwargs))


def nansum_complex(x, dim=-1, keepdim=False):
    # here we don't care, itshould be the case that both real and imag are nans
    return torch.complex(x.real.nansum(dim=dim, keepdim=keepdim), x.imag.nansum(dim=dim, keepdim=keepdim))


def tuple_to_device(t, device=torch.device("cpu")):
    if isinstance(t, torch.Tensor):
        return t.to(device=device)
    if t is None:
        return t
    return tuple(tuple_to_device(ti, device=device) for ti in t)


def awgn(original_signal, target_snr_db):
    """
    Adds white gaussian noise whose variance is adjusted to fit a given signal-to-noise ratio

    Parameters
    ----------
    original_signal : np.array
        original signal.
    target_snr_db : float
        target signal-to-noise ratio (in db).

    Returns
    -------
    noisy_signal : np.array
        noisy signal s.t. snr_db(noisy_signal, original_signal) = target_snr_db.

    """
    Px_db = signal_power_db(original_signal)
    var = db_to_power(Px_db - target_snr_db)
    std = torch.sqrt(var)
    if original_signal.dtype.is_complex:
        return complex_circular_gaussian_noise(original_signal, std)
    return torch.normal(original_signal, std)


def zero_pad(a, target_len, dim=-1):
    assert dim == -1
    return torch.nn.functional.pad(a, (0, target_len - a.shape[-1]), mode="constant", value=0)


def crop_or_zero_pad_to_target_len(a, target_len, dim=-1):
    # functional.pad seems to also be able to crop
    return zero_pad(a, target_len, dim=dim)


def fftconvolve_complex(x: torch.Tensor, y: torch.Tensor, mode: str = "full") -> torch.Tensor:
    r"""
    Same as torchaudio.functional.fftconvolve but for complex_valued tensors
    """
    _check_shape_compatible(x, y)
    _check_convolve_mode(mode)

    n = x.size(-1) + y.size(-1) - 1
    fresult = torch.fft.fft(x, n=n) * torch.fft.fft(y, n=n)
    result = torch.fft.ifft(fresult, n=n)
    return _apply_convolve_mode(result, x.size(-1), y.size(-1), mode)


def fftconvolve(x: torch.Tensor, y: torch.Tensor, mode: str = "full", dim=-1) -> torch.Tensor:
    """
    Wrapper around torchaudio.fftconvolve if the inputs aren't real'

    Parameters
    ----------
    x : torch.Tensor
        DESCRIPTION.
    y : torch.Tensor
        DESCRIPTION.
    mode : str, optional
        DESCRIPTION. The default is "full".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x_transposed = x.transpose(-1, dim)
    y_transposed = y.transpose(-1, dim)
    if x.dtype.is_complex or y.dtype.is_complex:
        res = fftconvolve_complex(x_transposed, y_transposed, mode)
    else:
        res = fftconvolve_torchaudio(x_transposed, y_transposed, mode)
    return res.transpose(-1, dim)


def toeplitz(x):
    # returns the toeplitz matrix which 1st row is x
    # res = torch.zeros(x.shape + (x.shape[-1],), device=x.device, dtype=x.dtype)
    # for i in range(x.shape[-1]):
    #     res[..., i:] = x[..., :i]
    res = torch.tril(torch.stack([torch.roll(x, shifts=i, dims=-1) for i in range(x.shape[-1])], dim=-1))
    return res


def test_toeplitz():
    import scipy

    x = torch.rand(512)
    x_toeplitz = toeplitz(x)
    x_scipy_toeplitz = torch.tensor(scipy.linalg.toeplitz(x.numpy(), torch.zeros_like(x).numpy()))
    assert torch.allclose(x_toeplitz, x_scipy_toeplitz), torch.dist(x_scipy_toeplitz, x_toeplitz)


def solve_lstsq_qr(A, y):
    # Does the same as torch.linalg.lstsq but faster backward
    Q, R = torch.linalg.qr(A, mode="reduced")
    rhs = Q.mH @ y.unsqueeze(-1)
    return torch.linalg.solve_triangular(R, rhs, upper=True).squeeze(-1)

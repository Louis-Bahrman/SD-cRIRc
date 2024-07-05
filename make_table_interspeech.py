#!/usr/bin/env python
# coding: utf-8


import torch

torch.manual_seed(0)


import os
import pickle

import numpy as np
import tqdm.auto as tqdm
from analyze_results import reinstantiate_cli
from datasets import (
    AudioDatasetConvolvedWithRirDataset,
    LibriSpeechAudioOnlyDataset,
    SynthethicRirDataset,
    WSJDataset,
    MedleyDBRawVoiceDataset,
)
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    ShortTimeObjectiveIntelligibility,
)
from IPython.display import clear_output
import pandas as pd
from model.convolutive_models import CTF, FourierDeconvolution
from model.physical_properties_estimators.edc import EDCdB
from model.physical_properties_estimators.edr import EDR
from model.losses import FilteredLoss
import torcheval.metrics


# In[132]:


REAL_TEST = True
NUM_BATCHES = float("inf")  # 10
mode = "speech"  # rir # Singing voice
# mode="rir"


def download(ckpt_name=None):
    if ckpt_name is None:
        ckpt_name = os.popen(
            "ssh gpu-gw ls /home/ids/lbahrman/dereverb-losses/lightning_logs/final/fcp_stft/seed6_0.1_3_noss_norecompute/version_5/checkpoints | sort -Vr | head -1"
        ).read()[:-1]
    print(ckpt_name)

    base_path = "/home/louis/dereverb-losses/lightning_logs/final/fcp_stft/seed6-localbackup/"
    version = "version_7"
    model_paths = {
        "FSN": os.path.join(base_path, "baseline", version),
        "+ SB": os.path.join(base_path, "FCP-vs-STFT", version),
        "+ CSB": os.path.join(base_path, "FCP-SS", version),
        "+ SSB": os.path.join(base_path, "FCP-vs-FCP", version),
        "+ 3B": os.path.join(base_path, "FCP-3-vs-STFT", version),
        "input": None,
    }

    cmd = " ".join(
        [
            "rsync",
            "-avPL",
            "--include='**/config.yaml'",
            '--include="' + ckpt_name + '"',
            "--include='*/'",
            "--exclude='*'",
            "gpu-gw:/home/ids/lbahrman/dereverb-losses/lightning_logs/final/fcp_stft/seed6/",
            "/home/louis/dereverb-losses/lightning_logs/final/fcp_stft/seed6-localbackup/",
        ]
    )
    print("downloading")
    os.system(cmd)
    return ckpt_name, model_paths


# In[136]:

if REAL_TEST:
    rirs_test_same = SynthethicRirDataset(rir_root="data/rirs_v2_test_same/")
    rirs_test_hard = SynthethicRirDataset(rir_root="data/rirs_v2_test_hard/")
    wsj_test = WSJDataset("data/speech", "test")
    librispeech_test_clean = LibriSpeechAudioOnlyDataset("data/speech/", "test-clean")
    librispeech_test_other = LibriSpeechAudioOnlyDataset("data/speech/", "test-other")
    medleydb_test = MedleyDBRawVoiceDataset()
else:
    rirs_test_same = SynthethicRirDataset(rir_root="data/rirs_v2_val_same/")
    rirs_test_hard = SynthethicRirDataset(rir_root="data/rirs_v2_val_hard/")
    wsj_test = WSJDataset("data/speech", "train")
    librispeech_test_clean = LibriSpeechAudioOnlyDataset("data/speech/", "dev-clean")
    librispeech_test_other = LibriSpeechAudioOnlyDataset("data/speech/", "dev-other")

dataset_args = dict(
    dry_signal_target_len=49151,
    rir_target_len=16383,
    align_and_scale_to_direct_path=True,
    dry_signal_start_index=16000,
    # dry_signal_start_index=None,
    convolve_here=True,
    normalize=True,
    pre_associate=False,
)


metrics_speech = {
    "STOI": ShortTimeObjectiveIntelligibility(fs=16000, extended=False),
    # "E-STOI": ShortTimeObjectiveIntelligibility(fs=16000, extended=True),
    "SISDR": ScaleInvariantSignalDistortionRatio(),
    "PESQ WB": PerceptualEvaluationSpeechQuality(fs=16000, mode="wb"),
    # "PESQ NB": PerceptualEvaluationSpeechQuality(fs=16000, mode="nb"),
}

metrics_singing_voice = {
    "FAD": torcheval.metrics.FrechetAudioDistance.with_vggish(),
    "SISDR": ScaleInvariantSignalDistortionRatio(),
}

deconvolvers_and_metrics_rir = {
    "1-band EDR": (
        FCP(
            num_absolute_cross_bands=0,
            output_len=64,
            crop_toeplitz=False,
            pad_X_hat_before_unfold=True,
            epsilon_lambda=1e0,
            # epsilon_lambda=1e-4, # Lower reconstruction SNR),
            return_stft=True,
            use_deconvolve_fourier=False,
            solve_qr=False,
        ),
        EDR(scale="bandwise"),
        FilteredLoss(base_loss=torch.nn.MSELoss(), min=-20, consider_only_target_mask=True),
    ),
    "3-band EDR": (
        FCP(
            num_absolute_cross_bands=1,
            output_len=64,
            crop_toeplitz=False,
            pad_X_hat_before_unfold=True,
            epsilon_lambda=1e0,
            # epsilon_lambda=1e-4, # Lower reconstruction SNR),
            return_stft=True,
            use_deconvolve_fourier=False,
            solve_qr=False,
        ),
        EDR(scale="bandwise"),
        FilteredLoss(base_loss=torch.nn.MSELoss(), min=-20, consider_only_target_mask=True),
    ),
    "Fourier EDC": (
        NonBlindRIREstimator(epsilon=1e-8),
        EDCdB(),
        FilteredLoss(base_loss=torch.nn.MSELoss(), min=-20, consider_only_target_mask=True),
    ),
}


if "speech" in mode.lower():
    datasets_noiseless = {
        "WSJ same": AudioDatasetConvolvedWithRirDataset(wsj_test, rirs_test_same, **dataset_args),
        "LibriSpeech clean same": AudioDatasetConvolvedWithRirDataset(
            librispeech_test_clean, rirs_test_same, **dataset_args
        ),
        "WSJ hard": AudioDatasetConvolvedWithRirDataset(wsj_test, rirs_test_hard, **dataset_args),
        "LibriSpeech clean hard": AudioDatasetConvolvedWithRirDataset(
            librispeech_test_clean, rirs_test_hard, **dataset_args
        ),
    }
    deconvolvers_and_metrics_rir = {}
    metrics_dry = metrics_speech
elif "sing" in mode.lower() or "voice" in mode.lower():
    datasets_noiseless = {
        "MedleyDB same": AudioDatasetConvolvedWithRirDataset(medleydb_test, rirs_test_same, **dataset_args),
        "MedleyDB hard": AudioDatasetConvolvedWithRirDataset(medleydb_test, rirs_test_hard, **dataset_args),
    }
    deconvolvers_and_metrics_rir = {}
    metrics_dry = metrics_singing_voice
elif "rir" in mode.lower():
    datasets_noiseless = {
        "WSJ same": AudioDatasetConvolvedWithRirDataset(wsj_test, rirs_test_same, **dataset_args),
        "WSJ hard": AudioDatasetConvolvedWithRirDataset(wsj_test, rirs_test_hard, **dataset_args),
    }
    metrics_dry = {}
else:
    raise ValueError(f"mode {mode} not allowed")
# datasets_noisy = {f"{k1} {snr_y} dB SNR": (v1, snr_y) for k1, v1 in datasets_noiseless.items() for snr_y in [0, 20]}
# datasets_noisy={k:v for k,v in datasets_noiseless.items() if "WSJ" in k}
all_datasets = datasets_noiseless


# In[138]:


def apply_func_to_dict(d, func=lambda l: np.array(l).mean()):
    if isinstance(d, dict):
        return {k: apply_func_to_dict(v) for k, v in d.items()}
    return func(d)


# In[139]:


def stringify(x, bold_mean=False, bold_std=None):
    if bold_std is None:
        bold_std = bold_mean
    a = np.array(x)
    str_mean = "{:.3f}".format(np.nanmean(a))
    str_std = "{:.3f}".format(np.nanstd(a))
    if bold_mean:
        str_mean = r"\bfseries " + str_mean
    if bold_std:
        str_std = r"\bfseries " + str_std
    return str_mean + " & " + str_std


def dict_to_latex_table(d, type="rir", mask=None):
    df_means = pd.DataFrame(d).applymap(lambda l: np.nanmean(np.array(l))).T
    if "rir" in type.lower():
        keys_of_interest = df_means.iloc[:-1].idxmin()
    else:
        keys_of_interest = df_means.idxmax().to_dict()
    s = ""
    for k1, v1 in d.items():
        s += k1
        for k2, v2 in v1.items():
            s += " & "
            bold = (k2, k1) in keys_of_interest.items()
            s += stringify(v2, bold_mean=bold)
        s += r" \\ " + " \n"
    return s


def test_model_dataset(
    model,
    dataset,
    dataset_name="",
    measure_input: bool = False,
    metrics_dry=metrics_dry,
    deconvolvers_and_metrics_rir=deconvolvers_and_metrics_rir,
):
    torch.manual_seed(0)
    if isinstance(dataset, tuple):
        dataset, snr_y = dataset
    else:
        snr_y = torch.inf
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=18,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )
    device = model.device if model is not None else torch.device("cuda")
    metrics_dry_good_device = {k: v.to(device=device) for k, v in metrics_dry.items()}
    res_dry = {dataset_name + " " + k: [] for k in metrics_dry.keys()}
    res_rir = {dataset_name + " " + k: [] for k in deconvolvers_and_metrics_rir.keys()}
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, leave=False)):
        if batch_idx > NUM_BATCHES:
            break
        # if model is None:
        #     y_noisy, (x, _, k) = batch
        #     for metrics_name, metric_fn in metrics_dry_good_device.items():
        #         res_speech[metrics_name].append(metric_fn(y_noisy[..., : x.size(-1)], x).cpu().item())
        #     return res_speech|res_rir
        batch = tuple_to_device(batch, device=device)
        y, (x, _, k) = batch
        y_noisy = awgn(y, snr_y)
        with torch.no_grad():
            if measure_input:
                x_hat = y_noisy
            else:
                x_hat = model.dereverberate_only(y_noisy)
            if len(deconvolvers_and_metrics_rir) > 0:
                X_hat = model.preproc.get_stft(x_hat)
                K = model.preproc.get_stft(k)
                X = model.preproc.get_stft(x)
                Y = model.preproc.get_stft(y)
                Y_noisy = model.preproc.get_stft(y_noisy)
                if measure_input:
                    x_hat = x
                    X_hat = X

        for metrics_name, metric_fn in metrics_dry_good_device.items():
            try:
                # FAD needs state update
                if "FAD" in metrics_name:
                    metric_fn.update(x_hat[..., 0, : x.size(-1)], x[:, 0, :])
                else:
                    res_dry[dataset_name + " " + metrics_name].append(
                        metric_fn(x_hat[..., : x.size(-1)], x).cpu().item()
                    )
            except Exception as e:
                print(e)
        for deconvolver_and_metric_name, (deconvolver, pp, pp_loss) in deconvolvers_and_metrics_rir.items():
            if getattr(deconvolver, "TIME_FREQUENCY_INPUT", False):
                K_hat = deconvolver(Y_noisy, X_hat)
                k_hat = model.postproc.invert_stft(K_hat)[..., : k.size(-1)]
                if K_hat.isnan().any():
                    ...
            else:
                k_hat = deconvolver(y, x_hat)
                K_hat = model.preproc.get_stft(k_hat)
            if getattr(pp, "TIME_FREQUENCY_INPUT", False):
                pp_est = pp(K_hat)
                pp_tgt = pp(K)
            else:
                pp_est = pp(k_hat)
                pp_tgt = pp(k)
            res = pp_loss(pp_est, pp_tgt).cpu().item()
            res_rir[dataset_name + " " + deconvolver_and_metric_name].append(res)

    # FAD needs compute in the end
    if "FAD" in metrics_dry_good_device.keys():
        res_dry[dataset_name + " " + "FAD"].append(metrics_dry_good_device["FAD"].compute().cpu().item())
        metrics_dry_good_device["FAD"].reset()
    return res_dry, res_rir


def test_models_and_datasets(
    model_paths,
    datasets,
    ckpt_name,
    metrics_dry,
    deconvolvers_and_metrics_rir,
):
    current_step = ckpt_name.split("=")[-1].split(".")[-2]
    res_dereverb, res_rirs = dict(), dict()
    for model_name, model_path in model_paths.items():
        if model_path is not None:
            cli = reinstantiate_cli(model_path, ckpt_name)
            device = torch.device("cuda")
            try:
                del model
            except:
                pass
            clear_output(wait=True)
            model = cli.model.to(device=device).eval()
        else:
            if len(deconvolvers_and_metrics_rir) == 0:
                model = None
        res_dereverb[model_name], res_rirs[model_name] = dict(), dict()
        for dataset_name, dataset in datasets.items():
            print(model_name, dataset_name)
            if len(metrics_dry) > 0:
                #                print("dereverb")
                #                print(dict_to_latex_table(res_dereverb, type="speech"))
                with open(str(current_step) + "_speech.pickle", mode="wb") as f:
                    pickle.dump(res_dereverb, f)
            if len(deconvolvers_and_metrics_rir) > 0:
                #                print("rirs")
                #                print(dict_to_latex_table(res_rirs, type="rir"))
                with open(str(current_step) + "_rirs.pickle", mode="wb") as f:
                    pickle.dump(res_rirs, f)
            print()
            res_dereverb_dataset, res_rirs_dataset = test_model_dataset(
                model,
                dataset,
                dataset_name=dataset_name,
                measure_input=model_path is None,
                metrics_dry=metrics_dry,
                deconvolvers_and_metrics_rir=deconvolvers_and_metrics_rir,
            )
            res_dereverb[model_name] |= res_dereverb_dataset
            res_rirs[model_name] |= res_rirs_dataset

    print("rirs")
    print(dict_to_latex_table(res_rirs, type="rir"))
    print("dereverb")
    print(dict_to_latex_table(res_dereverb, type="speech"))
    return res_rirs, res_dereverb


if __name__ == "__main__":
    # ckpt_name, model_paths = download()

    base_path = "./pretrained_models"
    version = "version_7"
    model_paths = {
        "FSN": os.path.join(base_path, "FSN", version),
        "+ SB": os.path.join(base_path, "FSN-SB", version),
        "+ CSB": os.path.join(base_path, "FSN-CSB", version),
        "+ SSB": os.path.join(base_path, "FSN-SSB", version),
        "+ 3B": os.path.join(base_path, "FSN-3B", version),
        "input": None,
    }
    ckpt_name = "epoch=21-step=330000.ckpt"
    test_models_and_datasets(
        model_paths=model_paths,
        datasets=all_datasets,
        ckpt_name=ckpt_name,
        metrics_dry=metrics_dry,
        deconvolvers_and_metrics_rir=deconvolvers_and_metrics_rir,
    )

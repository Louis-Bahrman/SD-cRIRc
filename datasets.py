#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:44:04 2023

@author: louis


Contient tout ce qu'il faut pour télécharger et générer les données de librispeech
Et les réverbérer avec pyroomacoustics

TODO faire de même avec wsjcam0 dans le cadre du reverb challenge  et WSJ

"""

import copy
import pyroomacoustics as pra
import numpy as np
import itertools
import torch
from torchaudio.datasets import LIBRISPEECH
import torchaudio.transforms
import lightning as L

import tqdm.auto as tqdm
import os
import pandas as pd
import soundfile as sf
import glob

from torch.utils.data import random_split
from model.utils.tensor_ops import crop_or_zero_pad_to_target_len, energy_to_db


class SynthethicRirDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rir_root: str = "./data/rirs_v2",
        num_new_rooms: int = 0,  # number of new rooms to generate
        room_dim_range: tuple[float, float] = (5.0, 10.0),
        room_height_range: tuple[float, float] = (2.5, 4.0),
        rt60_range: tuple[float, float] = (0.2, 1.0),
        num_sources_per_room: int = 1,
        num_mics_per_room: int = 16,
        min_distance_to_wall: float = 0.5,
        mic_height_range: tuple[float, float] = (0.7, 2),  # also used for source placement
        fs: int = int(16e3),
        query: str = "",
        return_properties: list[str] = ["rt_60"],  # properties to return
        source_mic_distance_range: tuple[float, float] | None = (0.75, 2.5),  # None for old behaviour
    ):
        # parse data to construct path
        self.rir_root = rir_root
        self.num_new_rooms = num_new_rooms
        self.room_dim_range = room_dim_range
        self.room_height_range = room_height_range
        self.rt60_range = rt60_range
        self.num_sources_per_room = num_sources_per_room
        self.num_mics_per_room = num_mics_per_room
        self.min_distance_to_wall = min_distance_to_wall
        self.mic_height_range = mic_height_range
        self.fs = fs
        self.query = query
        self.return_properties = return_properties
        self.source_mic_distance_range = source_mic_distance_range

        self.rir_properties = None
        self.filtered_rir_properties = None

        if self.num_new_rooms == 0:
            self._read_rir_csv()
            self._filter_rir_properties()

        if self.source_mic_distance_range is not None:
            # we force new behaviour
            self.num_mics_per_room = self.num_sources_per_room * self.num_mics_per_room
            self.num_sources_per_room = 1

    @property
    def rir_csv_path(self):
        return os.path.join(self.rir_root, "properties.csv")

    @property
    def num_filtered_rooms(self):
        return len(self.filtered_rir_properties["room_idx"].unique())

    @property
    def num_total_rooms(self):
        return len(self.rir_properties["room_idx"].unique())

    @property
    def num_filtered_rirs(self):
        return len(self.filtered_rir_properties.index)

    @property
    def num_rirs(self):
        return len(self.rir_properties.index)

    def _room_path(self, room_idx):
        return os.path.join(self.rir_root, f"room_{room_idx}")

    def _rir_path(self, room_idx, rir_idx_in_room):
        return os.path.join(self._room_path(room_idx), f"rir_{rir_idx_in_room}.wav")

    def _read_rir_csv(self):
        self.rir_properties = pd.read_csv(self.rir_csv_path, index_col="rir_global_idx")

    def _write_rir_csv(self):
        self.rir_properties.to_csv(self.rir_csv_path, float_format="%.3f")

    def _filter_rir_properties(self):
        if self.query is not None and self.query != "":
            self.filtered_rir_properties = self.rir_properties.query(self.query)
            # if len(self.filtered_rir_properties) == 0:
            #     raise ValueError("filtering returned empty dataset, try widening the search")
        else:
            self.filtered_rir_properties = self.rir_properties

    def generate_data_if_needed(self):
        if self.num_new_rooms > 0:
            self._generate_data()

    def _valid_position_range(self, shoebox_dim):
        return (
            [self.min_distance_to_wall, self.min_distance_to_wall, self.mic_height_range[0]],
            [*(shoebox_dim - self.min_distance_to_wall)[:2], self.mic_height_range[1]],
        )
        # return np.vstack((np.vstack((2 * [self.min_distance_to_wall], shoebox_dim[:2])).T, self.mic_height_range))
        #     zip(2 * [self.min_distance_to_wall], shoebox_dim[:2] - self.min_distance_to_wall, self.mic_height_range)
        # )

    def _sample_uniform_positions(self, shoebox_dim, num_positions=1):
        return self.rng.uniform(*self._valid_position_range(shoebox_dim), size=(num_positions, 3))

    def _is_valid_position(self, position, shoebox_dim):
        eps = 1e-5
        return (
            (self.min_distance_to_wall - eps <= position).all()
            and (position <= shoebox_dim - self.min_distance_to_wall + eps).all()
            and (self.mic_height_range[0] - eps <= position[..., -1]).all()
            and (position[..., -1] <= self.mic_height_range[1] + eps).all()
        )

    def _generate_data(self):
        self.rng = np.random.default_rng()
        os.makedirs(self.rir_root, exist_ok=True)
        if not os.path.isfile(self.rir_csv_path):
            # generate file

            self.rir_properties = pd.DataFrame(
                columns=[
                    "rir_global_idx",
                    "rir_path",
                    # Room properties
                    "room_idx",
                    "shoebox_length",
                    "shoebox_width",
                    "shoebox_height",
                    "volume",
                    "rt_60",
                    "absorption",
                    # rir_properties
                    "rir_idx_in_room",
                    "source_idx",
                    "mic_idx",
                    "source_x",
                    "source_y",
                    "source_z",
                    "mic_x",
                    "mic_y",
                    "mic_z",
                    "source_mic_distance",
                ]
            ).set_index("rir_global_idx")
            self._write_rir_csv()
            # self.rooms_properties=pd.DataFrame(columns=["room_idx","room_path", "shoebox_dim", "volume", "rt_60", "absorption"]).set_index("room_idx")
            # self._write_rooms_csv()
        # self._read_rooms_csv()
        self._read_rir_csv()
        current_num_new_rooms = 0
        rir_global_idx = self.rir_properties.index.max() + 1 if self.num_rirs > 0 else 1

        room_idx = self.rir_properties["room_idx"].max() + 1 if self.num_rirs > 0 else 1

        with tqdm.tqdm(total=self.num_new_rooms) as pbar:
            while current_num_new_rooms < self.num_new_rooms:
                shoebox_dim = np.zeros(3)
                shoebox_dim[:2] = self.rng.uniform(*self.room_dim_range, 2)
                shoebox_dim[2] = self.rng.uniform(*self.room_height_range)
                rt60 = self.rng.uniform(*self.rt60_range)
                try:
                    absorption, max_order = pra.inverse_sabine(rt60, shoebox_dim)
                except ValueError:
                    # Room too large for rt60
                    pass
                else:
                    room = pra.ShoeBox(
                        shoebox_dim, self.fs, absorption=absorption, max_order=max_order, use_rand_ism=True
                    )
                    volume = np.prod(room.shoebox_dim)
                    room_path = self._room_path(room_idx)
                    assert not os.path.isdir(room_path), "Error in room_idx generation"
                    # add sources
                    source_pos = self._sample_uniform_positions(shoebox_dim, num_positions=self.num_sources_per_room)
                    assert self._is_valid_position(source_pos, shoebox_dim)
                    for sp in source_pos:
                        room.add_source(sp)

                    # add mic old behaviour
                    if self.source_mic_distance_range is None:
                        # old behiaviour
                        mic_pos = self._sample_uniform_positions(shoebox_dim, num_positions=self.num_mics_per_room)
                        assert self._is_valid_position(mic_pos, shoebox_dim)
                        room.add_microphone_array(mic_pos.T)
                    # new behaviour
                    else:
                        mic_idx = 0
                        while mic_idx < self.num_mics_per_room:
                            mic_pos_within_room = False
                            source_mic_distance = self.rng.uniform(*self.source_mic_distance_range)
                            while not mic_pos_within_room:
                                normal_3d = self.rng.normal(size=3)
                                source_mic_vector = source_mic_distance * normal_3d / np.linalg.norm(normal_3d)
                                assert np.allclose(np.linalg.norm(source_mic_vector), source_mic_distance)
                                mic_pos = source_pos + source_mic_vector
                                mic_pos_within_room = self._is_valid_position(mic_pos, shoebox_dim)
                            mic_idx += 1
                            room.add_microphone(mic_pos.T)

                    room.compute_rir()

                    os.mkdir(room_path)
                    for rir_idx_in_room, (source_idx, mic_idx) in enumerate(
                        itertools.product(range(self.num_sources_per_room), range(self.num_mics_per_room))
                    ):
                        source_pos = room.sources[source_idx].position
                        mic_pos = room.mic_array.R.T[mic_idx]
                        source_mic_distance = np.linalg.norm(source_pos - mic_pos)
                        rir_path = self._rir_path(room_idx, rir_idx_in_room)
                        rir_properties_dict = {
                            "rir_path": rir_path,
                            # Room properties
                            "room_idx": room_idx,
                            "shoebox_length": shoebox_dim[0],
                            "shoebox_width": shoebox_dim[1],
                            "shoebox_height": shoebox_dim[2],
                            "volume": volume,
                            "rt_60": rt60,
                            "absorption": absorption,
                            # rir_properties
                            "rir_idx_in_room": rir_idx_in_room,
                            "source_idx": source_idx,
                            "mic_idx": mic_idx,
                            "source_x": source_pos[0],
                            "source_y": source_pos[1],
                            "source_z": source_pos[2],
                            "mic_x": mic_pos[0],
                            "mic_y": mic_pos[1],
                            "mic_z": mic_pos[2],
                            "source_mic_distance": source_mic_distance,
                        }

                        self.rir_properties.loc[rir_global_idx] = rir_properties_dict
                        rir = room.rir[mic_idx][source_idx]
                        sf.write(rir_path, rir, self.fs)
                        rir_global_idx += 1

                    room_idx += 1
                    current_num_new_rooms += 1
                    self._write_rir_csv()
                    pbar.update(1)

        self._write_rir_csv()
        self._filter_rir_properties()

    def __len__(self):
        return self.num_filtered_rirs

    def __getitem__(self, idx):
        # Use iloc
        rir_row = self.filtered_rir_properties.iloc[idx]
        rir_path = rir_row["rir_path"]
        other_properties = rir_row[self.return_properties]
        waveform, sample_rate = torchaudio.load(rir_path)
        if sample_rate != self.fs:
            raise ValueError(f"sample rate should be {self.fs}, but got {sample_rate}")
        return waveform, torch.tensor(other_properties).float()

    def random_split_by_rooms(self, *proportions):
        # Train test splits is implemented here since we use room information
        unique_room_idxs = self.filtered_rir_properties["room_idx"].unique()

        # We use torch random split function which is easier and also works with integers
        proportions = (1 - sum(proportions), *proportions)
        rooms_of_each_subset = random_split(unique_room_idxs, proportions)

        subsets = []
        # Only a shallow copy is needed, we will only modify the query method, not the dataframe
        for rooms_of_subset in rooms_of_each_subset:
            subset = copy.copy(self)
            subset.add_filter_to_query(f"room_idx.isin({list(rooms_of_subset)})")
            subset._filter_rir_properties()
            subsets.append(subset)
        return subsets

    def add_filter_to_query(self, filter_to_add):
        if self.query == "":
            self.query = filter_to_add
        else:
            self.query += " & " + filter_to_add


class LibriSpeechAudioOnlyDataset(LIBRISPEECH):
    """Same dataset as Librispeech, but returns audio only"""

    def __getitem__(self, idx):
        return super().__getitem__(idx)[0]


def normalize_sox(x: torch.Tensor, sample_rate: int = 16000):
    # dependant on samplerate so should be avoided
    return torchaudio.sox_effects.apply_effects_tensor(x, sample_rate=sample_rate, effects=[["norm"]])[0]


def normalize_max(x: torch.Tensor, target_max: float = 0.5):
    # 0.5 instead of sth ike 0.98 in order to not saturate when convolving with k
    return x / x.abs().max() * target_max


def remove_silent_windows(x: torch.Tensor, silence_power: float = -20.0, window_len: int = 1024):
    x_split = x.split(window_len, dim=-1)
    x_split_nonsilent = [t for t in x_split if energy_to_db(t.norm() ** 2) > silence_power]
    return torch.cat(x_split_nonsilent, dim=-1)


class AudioDatasetConvolvedWithRirDataset(torch.utils.data.Dataset):
    """For each audio, picks a random rir and convolve"""

    def __init__(
        self,
        audio_dataset,
        rir_dataset,
        dry_signal_target_len: int | None = 32767,
        rir_target_len: int | None = 16383,
        align_and_scale_to_direct_path=True,
        dry_signal_start_index=16000,  # None for random start
        pre_associate=False,
        convolve_here: bool = True,  # else convolve afterwards on GPU
        resampling_transform: torch.nn.Module | None = None,
        normalize: bool = True,
        ignore_silent_windows: bool = True,
    ):
        self.audio_dataset = audio_dataset
        self.rir_dataset = rir_dataset
        self.pre_associate = pre_associate and len(self.rir_dataset) > 0
        if self.pre_associate:
            self.pre_association = torch.randint(len(self.rir_dataset), size=(len(self.audio_dataset),))
        self.dry_signal_target_len = dry_signal_target_len
        self.rir_target_len = rir_target_len
        self.align_and_scale_to_direct_path = align_and_scale_to_direct_path
        self.dry_signal_start_index = dry_signal_start_index
        self.convolve_here = convolve_here
        self.resampling_transform = resampling_transform
        self.normalize = normalize
        self.normalize_op = normalize_max
        self.ignore_silent_windows = ignore_silent_windows
        assert self.normalize or not self.ignore_silent_windows, "need to normalize in order to ignore_silent_windows"

        if self.convolve_here:
            self.convolution_transform = torchaudio.transforms.FFTConvolve(mode="full")

    def __len__(self):
        return len(self.audio_dataset)

    def __getitem__(self, audio_idx):
        # Pick RIR
        if self.pre_associate:
            rir_idx = int(self.pre_association[audio_idx])
        else:
            rir_idx = int(torch.randint(len(self.rir_dataset), (1,)))
        rir, rir_properties = self.rir_dataset[rir_idx]

        # Pick dry
        x_full = self.audio_dataset[audio_idx]

        if self.normalize:
            x_full = self.normalize_op(x_full)
        if self.ignore_silent_windows:
            x_full = remove_silent_windows(x_full)

        if self.resampling_transform:
            x_full = self.resampling_transform(x_full)
            rir = self.resampling_transform(rir)

        if self.dry_signal_start_index is None:
            start_index = torch.randint(max(1, x_full.shape[-1] - self.dry_signal_target_len), size=(1,))[0]
        else:
            start_index = self.dry_signal_start_index
            if start_index >= x_full.size(-1):
                print("tensor not long enough to be cropped, skipping")
                return self.__getitem__(audio_idx + 1)
        x = x_full[..., start_index:]

        # Align and scale dry and RIR to direct path
        if self.align_and_scale_to_direct_path:
            peak_index = torch.argmax(torch.abs(rir))
            rir_peak = rir[..., peak_index]
            rir_aligned = rir[..., peak_index:]
            rir_aligned_scaled = rir_aligned / rir_peak
        else:
            rir_aligned_scaled = rir

        # Crop for linear convolution
        if self.dry_signal_target_len is not None:
            x_cropped = crop_or_zero_pad_to_target_len(x, target_len=self.dry_signal_target_len)
        else:
            x_cropped = x
        if self.rir_target_len is not None:
            rir_aligned_scaled_cropped = crop_or_zero_pad_to_target_len(
                rir_aligned_scaled, target_len=self.rir_target_len
            )
        else:
            rir_aligned_scaled_cropped = rir_aligned_scaled
        # Perform convolution on align rir and non scaled audio
        # We use x_cropped and not x_scaled_cropped for convolution because we don't want to scale 2 times
        if self.convolve_here:
            y = self.convolution_transform(rir_aligned_scaled_cropped, x_cropped)
        else:
            y = torch.full(
                x.shape[:-1] + (x_cropped.shape[-1] + rir_aligned_scaled_cropped.shape[-1] - 1,),
                fill_value=torch.nan,
            )
        return y, (x_cropped, rir_properties, rir_aligned_scaled_cropped)


class AudioDatasetConvolvedWithRirDatasetDataModule(L.LightningDataModule):
    def __init__(
        self,
        rir_dataset: torch.utils.data.Dataset,
        rir_dataset_test: torch.utils.data.Dataset | None = None,
        batch_size: int = 8,
        audio_root: str = "./data/speech",
        dry_signal_target_len: int = 32767,
        rir_target_len: int = 16383,
        align_and_scale_to_direct_path: bool = True,
        dry_signal_start_index_train: int | None = None,
        dry_signal_start_index_val_test: int | None = 16000,
        proportion_val_audio: float = 0.1,
        proportion_val_rir: float = 0.1,
        num_workers: int = 8,
        convolve_on_gpu: bool = False,
        resampling_transform: torch.nn.Module | None = None,
        num_distinct_rirs_per_batch: int | None = None,
        normalize: bool = True,
        ignore_silent_windows: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["rir_dataset", "rir_dataset_test"])
        self.save_hyperparameters(ignore=[], logger=False)

        self.rir_dataset = rir_dataset
        self.rir_dataset_test = rir_dataset_test
        if not os.path.isdir(self.hparams.audio_root):
            os.makedirs(self.hparams.audio_root)

        self.convolution_transform = torchaudio.transforms.FFTConvolve(mode="full")

    def prepare_data(self):
        self.rir_dataset.generate_data_if_needed()

    def setup(self, stage=None):
        # The splitting should be done in daughter class
        assert hasattr(self, "dry_train") and hasattr(self, "dry_val") and hasattr(self, "dry_test")
        self.generate_convolved_datasets()

    def generate_convolved_datasets(self):
        self.rir_dataset_train, self.rir_dataset_val = self.rir_dataset.random_split_by_rooms(
            self.hparams.proportion_val_rir
        )
        # Create the custom 'merged' datasets
        self.dataset_train = AudioDatasetConvolvedWithRirDataset(
            self.dry_train,
            self.rir_dataset_train,
            pre_associate=False,
            dry_signal_target_len=self.hparams.dry_signal_target_len,
            rir_target_len=self.hparams.rir_target_len,
            align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
            dry_signal_start_index=self.hparams.dry_signal_start_index_train,
            convolve_here=not self.hparams.convolve_on_gpu
            and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
            resampling_transform=self.hparams.resampling_transform,
            normalize=self.hparams.normalize,
            ignore_silent_windows=self.hparams.ignore_silent_windows,
        )
        self.dataset_val = AudioDatasetConvolvedWithRirDataset(
            self.dry_val,
            self.rir_dataset_val,
            pre_associate=True,
            dry_signal_target_len=self.hparams.dry_signal_target_len,
            rir_target_len=self.hparams.rir_target_len,
            align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
            dry_signal_start_index=self.hparams.dry_signal_start_index_val_test,
            convolve_here=not self.hparams.convolve_on_gpu
            and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
            resampling_transform=self.hparams.resampling_transform,
            normalize=self.hparams.normalize,
            ignore_silent_windows=self.hparams.ignore_silent_windows,
        )
        if self.rir_dataset_test is not None:
            self.dataset_test = AudioDatasetConvolvedWithRirDataset(
                self.dry_test,
                self.rir_dataset_test,
                pre_associate=True,
                dry_signal_target_len=self.hparams.dry_signal_target_len,
                rir_target_len=self.hparams.rir_target_len,
                align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
                dry_signal_start_index=self.hparams.dry_signal_start_index_val_test,
                convolve_here=not self.hparams.convolve_on_gpu
                and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
                resampling_transform=self.hparams.resampling_transform,
                normalize=self.hparams.normalize,
                ignore_silent_windows=self.hparams.ignore_silent_windows,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        assert self.rir_dataset_test is not None
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    @property
    def num_distinct_rirs_per_batch(self):
        if self.hparams.num_distinct_rirs_per_batch is None:
            return self.hparams.batch_size
        else:
            return self.hparams.num_distinct_rirs_per_batch

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch, (list, tuple)) and self.num_distinct_rirs_per_batch < self.hparams.batch_size:
            if self.hparams.batch_size % self.num_distinct_rirs_per_batch != 0:
                raise RuntimeError(
                    f"num_distinct_rirs_per_batch={self.num_distinct_rirs_per_batch} \
                        should divide batch size={batch.shape[0]}"
                )
            num_rirs_repeats = self.hparams.batch_size // self.num_distinct_rirs_per_batch
            batch[1][2] = batch[1][2][: self.num_distinct_rirs_per_batch, ...]
            batch[1][1] = batch[1][1][: self.num_distinct_rirs_per_batch, ...]
            if not self.hparams.convolve_on_gpu:
                batch[0] = self.convolution_transform(
                    batch[1][0], batch[1][2].repeat_interleave(num_rirs_repeats, dim=0)
                )
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # to perform convolution on gpu
        # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#on-after-batch-transfer
        if self.hparams.convolve_on_gpu and isinstance(batch, (list, tuple)):
            if self.num_distinct_rirs_per_batch < self.hparams.batch_size:
                num_rirs_repeats = self.hparams.batch_size // self.num_distinct_rirs_per_batch
                batch[0] = self.convolution_transform(
                    batch[1][0], batch[1][2].repeat_interleave(num_rirs_repeats, dim=0)
                )
            else:
                batch[0] = self.convolution_transform(batch[1][0], batch[1][2])
        # assert batch[0].isfinite().all()
        return batch


class LibrispeechSimulatedRirDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    def prepare_data(self):
        # Download librispeech selon config
        LibriSpeechAudioOnlyDataset(self.hparams.audio_root, url="train-clean-100", download=True)
        LibriSpeechAudioOnlyDataset(self.hparams.audio_root, url="test-clean", download=True)
        super().prepare_data()

    def setup(self, stage=None):
        # Pas besoin du stage, je n'ai rien de spécifique à certains stages (type transform ou augmentation)

        # split librispeech
        librispeech_train_full = LibriSpeechAudioOnlyDataset(
            self.hparams.audio_root, url="train-clean-100", download=False
        )
        proportions_librispeech = (1 - self.hparams.proportion_val_audio, self.hparams.proportion_val_audio)
        self.dry_train, self.dry_val = random_split(librispeech_train_full, proportions_librispeech)
        self.dry_test = LibriSpeechAudioOnlyDataset(self.hparams.audio_root, url="test-clean", download=False)
        super().setup(stage=stage)


class WSJDataset(torch.utils.data.Dataset):
    EXPECTED_SAMPLERATE = 16000
    TRAIN_TEST_DISKS = {
        "train": range(1, 13),
        "test": range(14, 16),
    }

    @property
    def wav_root(self):
        return os.path.join(self.audio_root, "WSJ", "WSJ0_wav_mic" + str(self.mic_number), self.subset)

    @property
    def sphere_root(self):
        return os.path.join(self.audio_root, "WSJ", "csr_1")

    @property
    def base_path(self):
        if self.wav:
            return self.wav_root
        else:
            return self.sphere_root

    def _check_base_path_exists(self):
        if not os.path.isdir(self.base_path):
            raise ValueError("Path does not exist or is not WSJ0")

    def __init__(self, audio_root, subset, mic_number: int = 1, wav: bool = True):
        self.audio_root = audio_root
        self.subset = subset
        self.mic_number = mic_number
        self.wav = wav
        self.paths_list = []

        self._check_base_path_exists()

        if self.wav:
            self.paths_list = glob.glob(os.path.join(self.base_path, "*.wav"))
        else:
            for i_disk in self.TRAIN_TEST_DISKS[subset]:
                self.paths_list.extend(
                    glob.glob(
                        os.path.join(self.base_path, "11-" + str(i_disk) + ".1", "**", "*.wv" + str(self.mic_number)),
                        recursive=True,
                    )
                )

    @property
    def len_hours(self):
        total_len = 0
        for path in tqdm.tqdm(self.paths_list):
            total_len += torchaudio.info(path).num_frames
        return total_len / self.EXPECTED_SAMPLERATE / 3600

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        path = self.paths_list[index]
        x, fs = torchaudio.load(path)
        assert fs == self.EXPECTED_SAMPLERATE
        return x

    def export_to_wav(self, new_path=None):
        if new_path is None:
            new_path = self.wav_root
        os.makedirs(new_path)
        for i in tqdm.trange((len(self))):
            x = self[i]
            torchaudio.save(os.path.join(new_path, f"{i}.wav"), x, self.EXPECTED_SAMPLERATE)


class MedleyDBRawVoiceDataset(torch.utils.data.Dataset):
    def __init__(self, fs=16000, base_path="data/music/MedleyDB_raw_voice"):
        self.base_path = base_path
        self.resampling_transform = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=16000,
            resampling_method="sinc_interp_kaiser",
        )
        self.paths_list = glob.glob(os.path.join(self.base_path, "*.wav"))

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        path = self.paths_list[index]
        x, fs = torchaudio.load(path)
        return self.resampling_transform(x)

    def extract_from_MedleyDB(
        self,
        original_path="/tsi/mir/MedleyDB/",
        singing_classes=["singer", "screamer", "rapper"],
    ):
        import shutil
        import yaml

        res_list = []
        song_dirs = os.listdir(original_path + "/Audio")
        for song_dir in song_dirs:
            print()
            print(song_dir)
            metadata_file = original_path + "/Metadata/" + song_dir + "_METADATA.yaml"
            with open(metadata_file) as f:
                metadata = yaml.load(f, yaml.loader.Loader)
            if "no" in metadata["has_bleed"]:
                raw_dir = metadata["raw_dir"]
                for v_stem in metadata["stems"].values():
                    for v_raw in v_stem["raw"].values():
                        for singing_class in singing_classes:
                            if singing_class in v_raw["instrument"]:
                                res_list.append(
                                    os.path.join(original_path, "Audio", song_dir, raw_dir, v_raw["filename"])
                                )
            print({"has_bleed": metadata["has_bleed"]})
        # with open("raw_songs", "w") as f:
        #     print("\n".join(res_list), file=f)
        for source_file in res_list:
            print("copying", source_file)
            shutil.copy2(source_file, self.base_path)


class MedleyDBRawVoiceRirDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    def setup(self, stage=None):
        self.dry_train = self.dry_val = self.dry_test = MedleyDBRawVoiceDataset
        super().setup(stage=stage)

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()


class WSJSimulatedRirDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    def setup(self, stage=None):
        self.wsj_train_full = WSJDataset(self.hparams.audio_root, subset="train")
        self.dry_test = WSJDataset(self.hparams.audio_root, subset="test")
        proportions_wsj = (1 - self.hparams.proportion_val_audio, self.hparams.proportion_val_audio)
        self.dry_train, self.dry_val = random_split(self.wsj_train_full, proportions_wsj)
        super().setup(stage=stage)


class DummyRirDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torchaudio.load("data/rirs/room_25/rir_4.wav")[0], torch.zeros(1)


class DummyDryDataset(torch.utils.data.Dataset):
    audios_list = [
        "data/speech/LibriSpeech/train-clean-100/6385/220959/6385-220959-002" + str(i) + ".flac" for i in range(1, 9)
    ]

    def __len__(self):
        return len(self.audios_list)

    def __getitem__(self, index):
        return torchaudio.load(self.audios_list[index])[0]


class DummyLibrispeechSimulatedRirDataModule(LibrispeechSimulatedRirDataModule):
    def prepare_data(self):
        # Download librispeech selon config
        pass

    def setup(self, stage=None):
        # Pas besoin du stage, je n'ai rien de spécifique à certains stages (type transform ou augmentation)
        self.dataset_val = AudioDatasetConvolvedWithRirDataset(
            DummyDryDataset(),
            DummyRirDataset(),
            pre_associate=True,
            dry_signal_target_len=self.hparams.dry_signal_target_len,
            rir_target_len=self.hparams.rir_target_len,
            align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
            dry_signal_start_index=self.hparams.dry_signal_start_index_val_test,
            convolve_here=not self.hparams.convolve_on_gpu
            and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
            resampling_transform=self.hparams.resampling_transform,
        )

    def train_dataloader(self):
        return self.val_dataloader()

    def test_dataloader(self):
        return self.val_dataloader()


def reset_batch_size(dataloader, new_batch_size):
    return torch.utils.data.DataLoader(
        dataset=dataloader.dataset,
        batch_size=new_batch_size,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
        pin_memory_device=dataloader.pin_memory_device,
    )


if __name__ == "__main__":
    # rir_dataset = SynthethicRirDataset(
    #     rir_root="./data/rirs_v2_val_hard",
    #     # num_new_rooms=500,
    #     num_new_rooms=10,
    #     room_dim_range=(10.0, 15.0),
    #     room_height_range=(4.0, 6.0),
    #     rt60_range=(1.0, 1.5),
    #     num_sources_per_room=1,
    #     num_mics_per_room=16,
    #     min_distance_to_wall=0.5,
    #     mic_height_range=(0.7, 3.5),  # also used for source placement
    #     source_mic_distance_range=(2.5, 4.0),
    # )
    # rir_dataset.generate_data_if_needed()
    rir_dataset = SynthethicRirDataset(
        rir_root="./data/rirs_v2_val_easy",
        # num_new_rooms=500,
        num_new_rooms=0,
    )
    rir_dataset.generate_data_if_needed()
    # data_module = WSJSimulatedRirDataModule(
    #     batch_size=8,
    #     rir_dataset=rir_dataset,
    #     dry_signal_start_index_train=None,
    #     proportion_val_audio=0.0,
    #     proportion_val_rir=0.0,
    # )
    # data_module.prepare_data()
    # data_module.setup()
    # loader = data_module.train_dataloader()
    # y, (x, rir_properties, rir) = next(iter(loader))

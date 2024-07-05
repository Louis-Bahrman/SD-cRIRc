#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:54:34 2023

@author: louis
"""

from lightning.pytorch.cli import LightningCLI
from model.full_model import FullModel
from datasets import AudioDatasetConvolvedWithRirDatasetDataModule


class MyCli(LightningCLI):
    def add_arguments_to_parser(self, parser):
        pass
        # parser.link_arguments("data.batch_size", "model.batch_size")


if __name__ == "__main__":
    cli = MyCli(
        model_class=FullModel,
        datamodule_class=AudioDatasetConvolvedWithRirDatasetDataModule,
        subclass_mode_model=False,
        subclass_mode_data=True,
    )

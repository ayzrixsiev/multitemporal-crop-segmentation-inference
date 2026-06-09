import json
import os
from datetime import datetime
from time import monotonic

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata


class Pastis_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        folds=None,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_data=None,
        sats=["S2"],
    ):
        """
        Pytorch dataset to load data from pastis, for semantic
        and panoptic segmentation.
        The dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series (how much time we took photo of this territory throught the year)
            - dates contains number of days passed since the image was made compared to reference date,
            we do this to solve "gap" problem, where model can see a clear amount of time passed from a previously taken image.
            we also can use this info to track growth speed, if a field turns from brown soil to bright green vegetation in 10 days, it’s likely one specific crop type.
            - target is the semantic or instance target
        Args:
            folder (str): path to the dataset
            norm (bool): if true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): it is used for temporal attention layer.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
            cache (bool): if true we save intially loaded images in ram for faster access.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                they are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): to create grouping of classes
            mono_date (int, str, optional): if you provide an argm, only one date out
                whole time-series data will be processed. if it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (Sentinel-2)
        """
        super(Pastis_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.target = target
        self.sats = sats
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        if mono_data is not None:
            self.mono_date = (
                datetime(*map(int, mono_data.split("-")))
                if "-" in mono_data
                else int(mono_data)
            )
        self.memory = {}  # ram system for fast data access
        self.memory_dates = (
            {}
        )  # days calculations, happens once in the first epoch and saved here
        self.class_mapping = (
            np.vectorize(
                lambda x: class_mapping[x]
            )  # in case if there are new IDs, go through each pixel in label images and swap old IDs to new ones
            if class_mapping is not None
            else class_mapping
        )

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

        # get metadata
        print("Processing metadata..")

        # open and sort the data in ascending order
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(
            range(-200, 600)
        )  # array of numbers, 200 days before reference date and 600 days after
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            # perse string dates and calculate amount of days passed from reference date
            for pid, date_seq in dates.items():
                if type(date_seq) == str:
                    date_seq = json.loads(date_seq)
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                """
                create timeline matrix
                days | 1 | 2 | 3 | ... | 800 |
                img1 | 0 | 1 | 0 | ... |  1  |
                img2 | 1 | 1 | 0 | ... |  0  |
                img3 and etc

                we need it to identify dates for all snapshots of each image
                for our attention based model, we can 
                """
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)

            # we create a fast lookup dictionary using that matrix for temporal positional encoding
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

        # Select only provided by the user folds for training
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]  # rows of the data for one epoch
        self.id_patches = (
            self.meta_patch.index
        )  # save space IDs in the array, each of that huge numbers can be accessed easily using sequential integer number, like 1,2,3..
        # pytorch needs this logic

        # Get precalculated normalization values (mean, std) and link them to the corresponding folds
        # {
        #  "Fold_1": {"mean": [400, 500, 600...], "std": [120, 130, 110...]},
        #  "Fold_2": {"mean": [410, 490, 610...], "std": [115, 135, 105...]}
        # }
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                    os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = None
        print("Dataset ready.")

    # train data size to measure one epoch
    def __len__(self):
        return self.len

    # get temporal positional encoding, [2, 4, 12, 18 ...] days passed for of all snapshots of all images
    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

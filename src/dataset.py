import json
import os
from datetime import datetime

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
        mono_date=None,
        sats=["S2"],
    ):
        """
        pytorch dataset to load data from pastis, for semantic and instance segmentation
        our input is 4D tensors (time, channels, height, width) and
        dataset outputs the following tumple: ((data, dates), target)
        args:
            folder (str): path to the dataset
            norm (bool): if true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): it is used for temporal attention layer.
            target (str): 'semantic' or 'instance', defines which type of target is.
                returned by the dataloader.
            cache (bool): if true we save intially loaded images in ram for faster access (created a variable).
            mem16 (bool): additional argument for cache, if True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                they are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): to create grouping of classes, we can change the ID assigned to a ceratin crop
                and assign it to another, for example.
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
        if mono_date is not None:
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
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

        # open and sort the data in ascending order, to have a consistent input data
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(
            range(-200, 600)
        )  # array of numbers, 200 days before reference date and 600 days after, making 800 days window that covers all images
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
                create timeline matrix/grid - 1 means we took a photo, 0 means we did not

                days | 1 | 2 | 3 | ... | 800 |
                ------------------------------
                img1 | 0 | 1 | 0 | ... |  1  |
                ------------------------------
                img2 | 1 | 1 | 0 | ... |  0  |
                ------------------------------
                img3 and etc

                we need it to identify dates for all snapshots of each image for our attention model
                """
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)

            # we create a fast lookup dictionary using that matrix for temporal positional encoding
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }  # when you call it, we get one row: self.date_tables["S2"][10001] = np.array([0, 0, 1, 0, 0, 0, 1, 0, ..., 0])

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

    # training data size to measure one epoch
    def __len__(self):
        return self.len

    # get temporal positional encoding, [2, 4, 12, 18 ...] days passed for of all snapshots for chosen territory
    # we check where the 1s are in the row in the timeline matrix
    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    # get a patch
    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # check whether we saved a data in ram, if not take it from dataset and convert it from numpy arrays into pytorch tensors
        if not self.cache or item not in self.memory.keys():
            data = {  # T x C x H x W arrays
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }
            data = {s: torch.from_numpy(a) for s, a in data.items()}

            # we can not subtract 1D tensor (precalculated mean) from 4D tensor (our images), hence we are adding dummy dimensions to make it look [1, channels, 1, 1]
            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                    / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }

            """
            pick the task: semantic or instance segmentation
                - semantic ("what this crop is" task): just loads 2D semantic maps (ground truth)
                - instance ("unique field separation" task): creates two empty matrices/canvas:
                        1. size = np.zeros((*instance_ids.shape, 2)) canvas with two pages
                        2. object_semantic_annotation = np.zeros(instance_ids.shape)
                        we take one ground truth of the current patch, and start going through it to identify three main things:
                        height, width and crop type of the field. so that we can take them and put into our canvases.
                        example ground truth:
                        0   0   0   0   0
                        0   5   5   5   5  <-- this 3x4 block is Field #5, height 3 and width 4
                        0   5   5   5   5
                        0   5   5   5   5
                        0   0   0   0   0

                        the loop goes throught this image and finds where pixel is 5, and then changes the values in the size canvas from 0
                        to corresponding ground truth's field dimensions, in our case (3, 4)

                        0   0   0   0   0
                        0   3   3   3   3  <-- height
                        0   3   3   3   3
                        0   3   3   3   3
                        0   0   0   0   0

                        0   0   0   0   0
                        0   4   4   4   4  <-- width  
                        0   4   4   4   4
                        0   4   4   4   4
                        0   0   0   0   0

                        and in object_semantic_annotation canvas, we find pixel with number 5 and change the value in canvas from 0 to it's crop type,
                        wheat (1) in our example

                        0   0   0   0   0
                        0   1   1   1   1  
                        0   1   1   1   1
                        0   1   1   1   1
                        0   0   0   0   0 

                        these three layers are created by us, but we also have other 4 layers from the dataset itself.
                        in the end we merge all 7 together into one tensor - (128x128x7)
                        7 present layers:
                            - heatmap: field-center proximity, the closer pixel to the center the hotter it is, we teach the model where the heart of the field is
                            - instance_ids: unique field IDs for separation
                            - pixel_to_object_mapping: fields are broken down into zones, we can idetify which zone each pixel belongs to
                            - size: h and w of the fields (we created)
                            - object_semantic_annotation: crop type of each field (we created)
                            - pixel_semantic_annotation: every individual pixel's crop type
                        each pixel in the resulting 3D array (ground truth) holds info from these 7 layers 
                        let's look at the example:
                        pixel A (located in the center of the field ID #5, size 3x4, zone #8)
                        pixel A[heatmap] = 1.0 (the hottest spot, because it is the center), we have a measurement on how far this pixel from the center
                        pixel A[instance_id] = #5 (belongs to the unique field #5)
                        pixel A[pixel_to_object_mapping] = #8 (specific zone pixel belongs to in the field #5)
                        pixel A[size h or w] = 3 or 4
                        pixel A[object_semantic_annotation] = 1 (wheat) what crop is in the field that this pixel belongs to
                        pixel A[pixel_semantic_annotation] = 1 (wheat) what crop type this specific pixel is      
            """
            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            # getting 7 layers
            elif self.target == "instance":
                heatmap = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                # even though instance ids track each field, sometimes those fields are separated by zones, this part tracks that
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                # create two canvases
                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                        ] = pixel_semantic_annotation[instance_ids == instance_id][
                            0
                        ]
                # merge every layer
                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()

            # faster training trick, we save loaded images into our RAM, we take processed image and 7 layers ground truth and put them into our dict self.memory
            if self.cache:
                if (
                    self.mem16
                ):  # if enabled we cut from 32 float bit down to 16 to save ram
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        # start using ram for getting the data after epoch 1, bacause we processed the whole dataset and stored it all in ram for faster access
        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # grab the current day array ([12, 14, 49]) that matches current image stack, and store it in self.memory_dates, if was not saved before, cache trick
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        """
        unpack dicts, since data, dates were saved as dict, like this:
        data = {
            "S2": torch.Tensor(30, 10, 128, 128),
            "S1": torch.Tensor(50, 4, 128, 128)
        }

        so we extract only tensor (30, 10, 128, 128) and array (50, 4, 128, 128)
        """
        if len(self.sats) == 1:
            data = data[self.sats[0]]
            dates = dates[self.sats[0]]

        """
        if you choose semantic segmentation, you will get:
        
        data: 4D tensor (time, channels, height, width) satellite image
        dates: 1D array (time) list of days each image was taken relative to the reference date
        target: 2D array (height, width) image with just crop types indentifying numbers 

        if you choose instance segmentation, you will get:
        
        data: 4D tensor (time, channels, height, width) satellite image
        dates: 1D array (time) list of days each image was taken relative to the reference date
        target: 3D array (height, width, 7 layers)  
        """
        return (data, dates), target


"""
process raw date 20180905 into
str(x)[:4] "2018" (Year)
str(x)[4:6] "09" (Month)
str(x)[6:] "05" (Day)

and then subtract from reference date
"""


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return d.values


# normilize image using precalculated math
def compute_norm_vals(folder, sat):
    norm_vals = {}
    for fold in range(1, 6):
        dt = Pastis_Dataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        means = []
        stds = []
        for i, b in enumerate(dt):
            print("{}/{}".format(i, len(dt)), end="\r")
            data = b[0][0][sat]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))

"""
Training-time data augmentation for the semantic track. Not upstream.
"""

import torch
import torch.utils.data as tdata


def _require_single_tensor(data, dates, who):
    """Raise unless the sample is single-satellite, i.e. tensors and not dicts."""
    if not torch.is_tensor(data) or not torch.is_tensor(dates):
        raise NotImplementedError(
            "{} expects a single-satellite sample (data and dates as tensors). "
            "Build the dataset with one entry in `sats`.".format(who)
        )


class SpatialAugment(tdata.Dataset):
    """
    Not upstream. Wraps a dataset and applies one random flip / quarter-turn per
    sample to every frame of the image stack and to the target. Training only.
    """

    def __init__(self, dataset):
        super(SpatialAugment, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _transform(x, hflip, vflip, k):
        """Apply one already-drawn transform to the last two axes of x."""
        if hflip:
            x = torch.flip(x, dims=[-1])
        if vflip:
            x = torch.flip(x, dims=[-2])
        if k:
            x = torch.rot90(x, k, dims=(-2, -1))
        return x

    def __getitem__(self, item):
        (data, dates), target = self.dataset[item]
        _require_single_tensor(data, dates, "SpatialAugment")

        if target.dim() != 2:
            raise NotImplementedError(
                "SpatialAugment only handles a semantic target of shape "
                "(H, W); got {}. An instance target is (H, W, 7), whose "
                "spatial axes are (-3, -2), not (-2, -1).".format(tuple(target.shape))
            )
        if data.shape[-2] != data.shape[-1]:
            raise ValueError(
                "Quarter-turns need a square patch; got {}x{}.".format(
                    data.shape[-2], data.shape[-1]
                )
            )

        # torch's RNG, not numpy's: each DataLoader worker gets its own stream.
        hflip = bool(torch.rand(()) < 0.5)
        vflip = bool(torch.rand(()) < 0.5)
        k = int(torch.randint(0, 4, ()))

        data = self._transform(data, hflip, vflip, k)
        target = self._transform(target, hflip, vflip, k)

        return (data, dates), target

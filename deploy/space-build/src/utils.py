import collections.abc
import re

import torch
from torch.nn import functional as F
from torch.utils import data

np_str_obj_array_pattern = re.compile(r"[SaUO]")

"""
we have a problem: our pytorch dataloader grabs random tensors and tries to glue them together to get one batch
but our tensors [time, channels, height, width] have different time series sequences, meaning one territory
can have 48 images taken throught the year and another may have 27, pytorch needs to have same dimensions across all tensors
we are trying to make one batch of

we implemented dynamic padding, which means we first measure the longest time series dimension in the current batch (pad_collate func)
and then check whether others align with it, if no we pad them with zeroes (pad_tensor)

in the end we output 5D tensor (batch, time, channels, height, width)
"""


def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(batch[0].size()))
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("format not managed : {}".format(elem_type))


# model parameters counter
def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Loading a trained U-TAE checkpoint and running one patch through it.

This module knows nothing about HTTP. It takes a sample (one patch: a stack of
Sentinel-2 images plus the day each one was taken), pushes it through the model,
and returns plain Python numbers and PNG bytes that `server.py` can hand to the
browser.
"""

import io
import json
import os
import sys
from argparse import Namespace
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import torch
from PIL import Image

# webapp/ sits one level below the repo root, and `from src import ...` only
# resolves when the repo root is on the import path. Running `python
# webapp/server.py` puts webapp/ there instead, so we add the root ourselves.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import model_utils  # noqa: E402

from webapp.pastis_meta import (  # noqa: E402
    BLUE_BAND,
    GREEN_BAND,
    NIR_BAND,
    NUM_CLASSES,
    PIXEL_AREA_HA,
    RED_BAND,
    VOID_CLASS,
)


# --------------------------------------------------------------------------- #
#  Loading the trained run
# --------------------------------------------------------------------------- #


class ModelBundle:
    """A loaded checkpoint plus everything that describes the run it came from."""

    def __init__(self, model, config, device, fold, res_dir, meta):
        self.model = model
        self.config = config
        self.device = device
        self.fold = fold
        self.res_dir = res_dir
        self.meta = meta


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_bundle(res_dir, fold, device=None, weights=None):
    """Rebuild the architecture from conf.json and pour the trained weights in.

    A checkpoint is only a bag of numbers -- it does not describe the network
    those numbers belong in. conf.json, written by train_semantic.py at the start
    of the run, is what tells us the shape to rebuild.
    """
    res_dir = os.path.abspath(res_dir)
    conf = _read_json(os.path.join(res_dir, "conf.json"))
    if conf is None:
        raise FileNotFoundError(
            "No conf.json in {}. Point --res_dir at the results folder "
            "produced by train_semantic.py.".format(res_dir)
        )

    fold_dir = os.path.join(res_dir, "Fold_{}".format(fold))
    weights = weights or os.path.join(fold_dir, "model.pth.tar")
    if not os.path.exists(weights):
        raise FileNotFoundError("No checkpoint at {}".format(weights))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model_utils.get_model(Namespace(**conf), mode="semantic")
    state = torch.load(weights, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    # Checkpoints saved from a DataParallel-wrapped model carry a "module." prefix.
    if any(k.startswith("module.") for k in sd):
        sd = OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
    model.load_state_dict(sd)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta = {
        "fold": fold,
        "device": str(device),
        "n_params": n_params,
        "weights": weights,
        "epochs_trained": conf.get("epochs"),
        "batch_size": conf.get("batch_size"),
        "lr": conf.get("lr"),
        "ref_date": conf.get("ref_date", "2018-09-01"),
        "test_metrics": _read_json(os.path.join(fold_dir, "test_metrics.json")),
        "trainlog": _read_json(os.path.join(fold_dir, "trainlog.json")),
        "checkpoint_epoch": state.get("epoch"),
    }
    return ModelBundle(model, conf, device, fold, res_dir, meta)


# --------------------------------------------------------------------------- #
#  Loading a sample
# --------------------------------------------------------------------------- #


class Sample:
    """One patch: the image stack, when each image was taken, and optional extras.

    `raw` keeps the untouched reflectance values so we can still draw a true
    colour picture after the model has seen the normalised version.
    """

    def __init__(self, raw, dates, mean, std, **extra):
        self.raw = raw.astype(np.float32)  # (T, 10, H, W)
        self.dates = np.asarray(dates).astype(np.int64)  # (T,) days from ref
        self.mean = np.asarray(mean, dtype=np.float32)  # (10,)
        self.std = np.asarray(std, dtype=np.float32)  # (10,)
        self.target = extra.get("target")
        self.patch_id = extra.get("patch_id")
        self.fold = extra.get("fold")
        self.tile = extra.get("tile")
        self.bounds = extra.get("bounds")
        self.polygon = extra.get("polygon")
        self.ref_date = extra.get("ref_date", "2018-09-01")
        self.name = extra.get("name", "sample")
        self.warnings = list(extra.get("warnings", []))

    @property
    def n_dates(self):
        return int(self.raw.shape[0])

    def normalised(self):
        m = self.mean[None, :, None, None]
        s = self.std[None, :, None, None]
        return (self.raw - m) / s

    def calendar(self):
        ref = datetime(*map(int, str(self.ref_date).split("-")))
        return [(ref + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in self.dates]


def _first_present(npz, *names):
    for n in names:
        if n in npz.files:
            return npz[n]
    return None


def load_sample(path_or_bytes, name="sample", fallback_norm=None):
    """Read a sample from a .npz bundle or a bare PASTIS .npy stack.

    Everything except the image stack itself is optional. When a field is
    missing we fill in a sensible stand-in and record a warning so the interface
    can tell the user the result is approximate rather than silently lying.
    """
    warnings = []

    if isinstance(path_or_bytes, (bytes, bytearray)):
        handle = io.BytesIO(path_or_bytes)
        is_npz = bytes(path_or_bytes[:2]) == b"PK"
    else:
        handle = path_or_bytes
        is_npz = str(path_or_bytes).lower().endswith(".npz")

    if is_npz:
        npz = np.load(handle, allow_pickle=True)
        raw = _first_present(npz, "data", "S2", "x")
        if raw is None:
            raise ValueError("The .npz has no 'data' array (T x 10 x 128 x 128).")
        dates = _first_present(npz, "dates", "positions")
        mean = _first_present(npz, "mean")
        std = _first_present(npz, "std")
        target = _first_present(npz, "target", "truth")
        extra = {}
        for key in ("patch_id", "fold", "tile", "ref_date"):
            val = _first_present(npz, key)
            if val is not None:
                extra[key] = val.item() if getattr(val, "ndim", 1) == 0 else val
        for key in ("bounds", "polygon"):
            val = _first_present(npz, key)
            if val is not None:
                extra[key] = np.asarray(val).tolist()
    else:
        raw = np.load(handle)
        dates = mean = std = target = None
        extra = {}

    raw = np.asarray(raw)
    if raw.ndim != 4 or raw.shape[1] != 10:
        raise ValueError(
            "Expected a (T, 10, H, W) stack, got {}. This model reads a whole "
            "year of 10-band Sentinel-2 images, not a single photo.".format(
                tuple(raw.shape)
            )
        )

    if dates is None:
        # No calendar shipped with the file. Spread the acquisitions evenly over
        # one growing season so the positional encoding still gets a sane signal.
        dates = np.linspace(0, 364, raw.shape[0]).round().astype(np.int64)
        warnings.append(
            "No acquisition dates in the file. Assumed evenly spaced dates across "
            "one season -- the prediction is approximate."
        )

    if mean is None or std is None:
        if fallback_norm is not None:
            mean, std = fallback_norm
            warnings.append(
                "No normalisation values in the file. Used the dataset-wide "
                "statistics from NORM_S2_patch.json."
            )
        else:
            mean = raw.mean(axis=(0, 2, 3))
            std = raw.std(axis=(0, 2, 3)) + 1e-6
            warnings.append(
                "No normalisation values available. Fell back to this patch's own "
                "statistics, which is not what the model was trained with -- "
                "expect degraded accuracy."
            )

    if target is not None:
        target = np.asarray(target).astype(np.int64)
        if target.ndim == 3:  # instance-style target: first plane is the class map
            target = target[..., -1].astype(np.int64)

    extra["target"] = target
    extra["name"] = name
    extra["warnings"] = warnings
    return Sample(raw, dates, mean, std, **extra)


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #


def _label_components(mask):
    """Count separate blobs in a boolean map (4-connected flood fill).

    Two pixels belong to the same field only if you can walk between them going
    up, down, left or right without leaving the mask. Used to turn "42% of this
    patch is meadow" into "there are 7 separate meadows here".
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes = []
    current = 0
    for start_y in range(h):
        for start_x in range(w):
            if not mask[start_y, start_x] or labels[start_y, start_x]:
                continue
            current += 1
            size = 0
            stack = [(start_y, start_x)]
            labels[start_y, start_x] = current
            while stack:
                y, x = stack.pop()
                size += 1
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
            sizes.append(size)
    return labels, sizes


def _confusion(pred, truth, n=NUM_CLASSES):
    valid = truth != VOID_CLASS
    p = pred[valid].astype(np.int64)
    t = truth[valid].astype(np.int64)
    keep = (p >= 0) & (p < n) & (t >= 0) & (t < n)
    return np.bincount(t[keep] * n + p[keep], minlength=n * n).reshape(n, n)


def compute_stats(pred, conf, truth, min_field_px=4):
    """Everything the right-hand panel shows, per class and overall."""
    total_px = pred.size
    counts = np.bincount(pred.ravel(), minlength=NUM_CLASSES)

    cm = _confusion(pred, truth) if truth is not None else None
    per_class = []
    for c in range(NUM_CLASSES):
        # Void never appears in a score. train_semantic.py zeroes its loss weight
        # (`weights[config.ignore_index] = 0`, and -1 indexes the last class) and
        # deletes its row and column from the confusion matrix before computing
        # mIoU, so leaving it in here would not match the reported test number.
        if c == VOID_CLASS:
            continue
        n_px = int(counts[c])
        if n_px == 0 and (cm is None or cm[c].sum() == 0):
            continue
        mask = pred == c
        _, sizes = _label_components(mask)
        fields = [s for s in sizes if s >= min_field_px]
        entry = {
            "class_id": c,
            "pixels": n_px,
            "area_ha": round(n_px * PIXEL_AREA_HA, 2),
            "share": round(100.0 * n_px / total_px, 2),
            "confidence": round(float(conf[mask].mean() * 100), 1) if n_px else None,
            "fields": len(fields),
            "mean_field_ha": (
                round(float(np.mean(fields)) * PIXEL_AREA_HA, 2) if fields else None
            ),
            "largest_field_ha": (
                round(float(np.max(fields)) * PIXEL_AREA_HA, 2) if fields else None
            ),
        }
        if cm is not None:
            tp = float(cm[c, c])
            union = float(cm[c].sum() + cm[:, c].sum() - cm[c, c])
            entry["iou"] = round(100.0 * tp / union, 1) if union > 0 else None
            entry["truth_pixels"] = int(cm[c].sum())
            entry["truth_area_ha"] = round(cm[c].sum() * PIXEL_AREA_HA, 2)
        per_class.append(entry)

    per_class.sort(key=lambda e: e["pixels"], reverse=True)

    overall = {
        "labelled_area_ha": round(total_px * PIXEL_AREA_HA, 2),
        "classes_found": int((counts > 0).sum()),
        "total_fields": int(sum(e["fields"] for e in per_class)),
        "mean_confidence": round(float(conf.mean() * 100), 1),
        "low_confidence_share": round(float((conf < 0.5).mean() * 100), 1),
        "dominant_class": int(np.argmax(counts)),
    }
    if cm is not None:
        valid = truth != VOID_CLASS
        overall["accuracy"] = round(float((pred == truth)[valid].mean() * 100), 2)
        ious = [e["iou"] for e in per_class if e.get("iou") is not None]
        overall["miou"] = round(float(np.mean(ious)), 2) if ious else None
        overall["void_share"] = round(float((~valid).mean() * 100), 2)
    return per_class, overall, cm


def ndvi_map(raw):
    """Greenness for every date: near-infrared against red, pixel by pixel.

    Healthy leaves bounce back a lot of infrared and swallow red, so a growing
    crop scores high and bare soil scores low. Used to draw the greenness view.
    """
    nir = raw[:, NIR_BAND].astype(np.float32)
    red = raw[:, RED_BAND].astype(np.float32)
    return (nir - red) / np.clip(nir + red, 1e-6, None)


def cloud_flags(raw):
    """Rough guess at which dates are cloudy, from how bright the blue band is.

    Clouds are white, and white means a very high blue reading. A date whose blue
    channel sits far above the season's usual level is probably clouded over.
    This is a display hint, not a real cloud mask.
    """
    blue = raw[:, BLUE_BAND].mean(axis=(1, 2))
    median = float(np.median(blue))
    spread = float(np.median(np.abs(blue - median))) or 1.0
    score = (blue - median) / (1.4826 * spread)
    return [round(float(s), 2) for s in score], [bool(s > 2.5) for s in score]


# --------------------------------------------------------------------------- #
#  Picture rendering
# --------------------------------------------------------------------------- #

_NDVI_STOPS = [
    (0.00, (120, 96, 74)),
    (0.20, (176, 158, 116)),
    (0.35, (203, 199, 122)),
    (0.50, (140, 186, 96)),
    (0.70, (60, 145, 72)),
    (1.00, (14, 79, 46)),
]


def _ndvi_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    xs = np.linspace(-0.2, 1.0, 256)
    for i, x in enumerate(xs):
        t = float(np.clip((x + 0.2) / 1.2, 0, 1))
        for (a, ca), (b, cb) in zip(_NDVI_STOPS, _NDVI_STOPS[1:]):
            if a <= t <= b:
                f = (t - a) / max(b - a, 1e-6)
                lut[i] = [int(ca[k] + (cb[k] - ca[k]) * f) for k in range(3)]
                break
        else:
            lut[i] = _NDVI_STOPS[-1][1]
    return lut


NDVI_LUT = _ndvi_lut()


def render_sheet(raw, ndvi):
    """Pack every date into one wide PNG so the browser fetches it once.

    Row 0 is true colour, row 1 is the greenness map. The date slider then just
    crops a 128x128 window out of an image that is already in memory, which is
    what makes scrubbing through the year feel instant.
    """
    t, _, h, w = raw.shape
    rgb = raw[:, [RED_BAND, GREEN_BAND, BLUE_BAND]].astype(np.float32)

    # One brightness scale for the whole series, so seasonal change is real change
    # and not the contrast stretch moving under your feet.
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (rgb ** (1 / 1.15) * 255).astype(np.uint8)  # gentle lift on the shadows
    rgb = np.transpose(rgb, (0, 2, 3, 1))  # (T, H, W, 3)

    idx = np.clip(((ndvi + 0.2) / 1.2 * 255), 0, 255).astype(np.uint8)
    green = NDVI_LUT[idx]  # (T, H, W, 3)

    sheet = np.zeros((h * 2, w * t, 3), dtype=np.uint8)
    for i in range(t):
        sheet[0:h, i * w : (i + 1) * w] = rgb[i]
        sheet[h : 2 * h, i * w : (i + 1) * w] = green[i]

    buf = io.BytesIO()
    Image.fromarray(sheet).save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def encode_map(arr):
    """A 128x128 label or confidence map as a raw PNG the browser can read back.

    Sent as a greyscale image rather than a JSON array of 16384 numbers: it is
    about ten times smaller and the browser decodes it natively.
    """
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  The prediction itself
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict(bundle, sample):
    """Run one patch through U-TAE and gather everything worth showing."""
    x = torch.from_numpy(sample.normalised()).float().unsqueeze(0).to(bundle.device)
    d = torch.from_numpy(sample.dates).long().unsqueeze(0).to(bundle.device)

    logits, att = bundle.model(x, batch_positions=d, return_att=True)
    probs = torch.softmax(logits, dim=1)[0]  # (20, H, W)
    conf_t, pred_t = probs.max(dim=0)

    pred = pred_t.cpu().numpy().astype(np.uint8)
    conf = conf_t.cpu().numpy().astype(np.float32)
    truth = sample.target

    per_class, overall, cm = compute_stats(pred, conf, truth)
    ndvi = ndvi_map(sample.raw)
    scores, cloudy = cloud_flags(sample.raw)

    # att is (n_head, B, T, h, w) at the bottleneck resolution. Averaging away
    # height and width leaves "how much did the model lean on each date"; keeping
    # them but averaging the heads leaves "where did it look on that date".
    att = att[:, 0].float().cpu().numpy()  # (n_head, T, h, w)
    att_time = att.mean(axis=(2, 3))  # (n_head, T)
    att_space = att.mean(axis=0)  # (T, h, w)
    att_space = att_space / np.clip(att_space.max(), 1e-9, None)

    payload = {
        "name": sample.name,
        "patch_id": int(sample.patch_id) if sample.patch_id is not None else None,
        "fold": int(sample.fold) if sample.fold is not None else None,
        "tile": str(sample.tile) if sample.tile is not None else None,
        "n_dates": sample.n_dates,
        "dates": sample.calendar(),
        "day_offsets": [int(v) for v in sample.dates],
        "has_truth": truth is not None,
        "per_class": per_class,
        "overall": overall,
        "cloud_score": scores,
        "cloudy": cloudy,
        "attention_time": [[round(float(v), 6) for v in row] for row in att_time],
        "attention_space": [
            [round(float(v), 4) for v in frame.ravel()] for frame in att_space
        ],
        "attention_shape": list(att_space.shape[1:]),
        "bounds": sample.bounds,
        "polygon": sample.polygon,
        "warnings": sample.warnings,
        "confusion": cm.tolist() if cm is not None else None,
    }

    maps = {
        "pred": encode_map(pred),
        "conf": encode_map(np.clip(conf * 255, 0, 255)),
        "sheet": render_sheet(sample.raw, ndvi),
    }
    if truth is not None:
        maps["truth"] = encode_map(truth)
        maps["error"] = encode_map(
            np.where((pred != truth) & (truth != VOID_CLASS), 255, 0)
        )
    return payload, maps

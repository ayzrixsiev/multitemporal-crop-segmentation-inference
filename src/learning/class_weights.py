"""
Class weights for the semantic loss, derived from label frequency. Not upstream.
"""

import json
import os

import numpy as np
import torch

CACHE_NAME = "class_counts.json"


def count_label_pixels(folder, id_patches, num_classes, display_step=100):
    """
    Count pixels per class over a set of patches, reading only the annotations.

    Returns a (num_classes,) int64 array.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    total = len(id_patches)

    for i, id_patch in enumerate(id_patches):
        path = os.path.join(folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
        labels = np.asarray(np.load(path, mmap_mode="r")[0], dtype=np.int64)

        if labels.max() >= num_classes or labels.min() < 0:
            raise ValueError(
                "Patch {} has label ids outside [0, {}): min {} max {}. "
                "num_classes and the annotations disagree.".format(
                    id_patch, num_classes, labels.min(), labels.max()
                )
            )

        counts += np.bincount(labels.ravel(), minlength=num_classes)

        if display_step and (i + 1) % display_step == 0:
            print("  counting labels {}/{}".format(i + 1, total), end="\r")

    if display_step:
        print("  counting labels {}/{} done.".format(total, total))
    return counts


def load_or_compute_counts(res_dir, folder, id_patches, folds, num_classes):
    """
    Per-class pixel counts for a set of folds, cached in res_dir.

    The cache is keyed by the sorted fold list, so another fold set never reuses
    these counts. Returns a (num_classes,) int64 array.
    """
    key = "-".join(str(f) for f in sorted(folds))
    cache_path = os.path.join(res_dir, CACHE_NAME)

    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as file:
                cache = json.loads(file.read())
        except (ValueError, OSError):
            cache = {}

    entry = cache.get(key)
    if (
        entry is not None
        and entry.get("num_classes") == num_classes
        and entry.get("n_patches") == len(id_patches)
    ):
        print("Class counts for folds {} read from {}".format(key, cache_path))
        return np.asarray(entry["counts"], dtype=np.int64)

    print(
        "Counting class pixels over folds {} ({} patches, labels only)...".format(
            key, len(id_patches)
        )
    )
    counts = count_label_pixels(folder, id_patches, num_classes)

    cache[key] = {
        "num_classes": num_classes,
        "n_patches": len(id_patches),
        "counts": [int(c) for c in counts],
    }
    with open(cache_path, "w") as file:
        file.write(json.dumps(cache, indent=4))
    print("Class counts cached in {}".format(cache_path))
    return counts


def keep_mask(num_classes, ignore_index):
    """
    Boolean array that is False exactly at the ignored class.

    Lets ignore_index = -1 and ignore_index = 19 behave identically.
    """
    keep = np.ones(num_classes, dtype=bool)
    if ignore_index is not None:
        keep[ignore_index] = False
    return keep


def build_class_weights(counts, mode, num_classes, ignore_index):
    """
    Turn pixel counts into loss weights, one of none / inverse / sqrt_inverse.

    The ignored class ends at weight 0 and the rest are rescaled to mean 1.
    Returns a (num_classes,) float64 array.
    """
    counts = np.asarray(counts, dtype=np.float64)
    keep = keep_mask(num_classes, ignore_index)

    if mode == "none":
        weights = np.ones(num_classes, dtype=np.float64)
        weights[~keep] = 0.0
        return weights

    # Clamp to one pixel so a class with no pixels gives the largest finite
    # weight rather than 1/0.
    safe = np.clip(counts, 1.0, None)
    freq = safe / safe[keep].sum()

    if mode == "inverse":
        weights = 1.0 / freq
    elif mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(freq)
    else:
        raise ValueError(
            "Unknown class_weights mode {!r}, expected one of "
            "none / inverse / sqrt_inverse".format(mode)
        )

    weights[~keep] = 0.0
    weights[keep] = weights[keep] * (keep.sum() / weights[keep].sum())

    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Class weights came out non-finite or negative.")
    return weights


def print_weight_table(counts, weights, num_classes, ignore_index):
    """Print one line per class: pixel count, share of scored pixels, weight."""
    counts = np.asarray(counts, dtype=np.float64)
    keep = keep_mask(num_classes, ignore_index)
    total = counts[keep].sum()

    print("Class weights:")
    print("  {:>3}  {:>14}  {:>8}  {:>8}".format("id", "pixels", "share%", "weight"))
    for c in range(num_classes):
        share = 100.0 * counts[c] / total if total else float("nan")
        tag = "  (ignored)" if not keep[c] else ""
        print(
            "  {:>3}  {:>14,}  {:>8.2f}  {:>8.3f}{}".format(
                c, int(counts[c]), share, weights[c], tag
            )
        )
    kept = weights[keep]
    if kept.size:
        print(
            "  spread {:.2f}x  (min {:.3f}, max {:.3f})".format(
                kept.max() / kept.min() if kept.min() > 0 else float("inf"),
                kept.min(),
                kept.max(),
            )
        )


def class_weights_for_training(config, dataset, folds, device):
    """
    Build the loss weight tensor for one fold from --class_weights.

    With "none" this returns upstream's torch.ones with a 0 at the ignored
    class and reads nothing from disk. Returns a (num_classes,) float tensor.
    """
    weights = np.ones(config.num_classes, dtype=np.float64)
    keep = keep_mask(config.num_classes, config.ignore_index)
    weights[~keep] = 0.0

    if config.class_weights != "none":
        if getattr(dataset, "class_mapping", None) is not None:
            raise NotImplementedError(
                "--class_weights counts raw label ids. The dataset was built "
                "with a class_mapping, so the counts would not match the "
                "classes the model predicts."
            )
        counts = load_or_compute_counts(
            res_dir=config.res_dir,
            folder=config.dataset_folder,
            id_patches=list(dataset.id_patches),
            folds=folds,
            num_classes=config.num_classes,
        )
        weights = build_class_weights(
            counts,
            mode=config.class_weights,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )
        print_weight_table(counts, weights, config.num_classes, config.ignore_index)

    return torch.from_numpy(weights).float().to(device)

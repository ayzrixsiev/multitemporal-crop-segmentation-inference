import argparse
import json
import os
import shutil

import geopandas as gpd
import numpy as np

FOLD_SEQUENCE = [
    [[1, 2, 3], [4], [5]],
    [[2, 3, 4], [5], [1]],
    [[3, 4, 5], [1], [2]],
    [[4, 5, 1], [2], [3]],
    [[5, 1, 2], [3], [4]],
]

VOID = 19
REF_DATE = "2018-09-01"

GROUPS = [
    {"id": 1, "dir": "group1", "label": "Group 1"},
    {"id": 2, "dir": "group2", "label": "Group 2"},
]


def load_norm(folder, folds):
    with open(os.path.join(folder, "NORM_S2_patch.json"), "r") as fh:
        vals = json.load(fh)
    means = np.stack([vals["Fold_{}".format(f)]["mean"] for f in folds])
    stds = np.stack([vals["Fold_{}".format(f)]["std"] for f in folds])
    return means.mean(axis=0).astype(np.float32), stds.mean(axis=0).astype(np.float32)


def to_wgs84(meta):
    if meta.crs is None:
        meta = meta.set_crs(2154, allow_override=True)
    try:
        return meta.to_crs(4326)
    except Exception as exc:
        print(
            "Could not re-project geometry ({}); bundles will have no map.".format(exc)
        )
        return None


def date_strings(raw):
    seq = json.loads(raw) if isinstance(raw, str) else raw
    values = seq.values() if isinstance(seq, dict) else seq
    return sorted(
        "{}-{}-{}".format(str(int(v))[:4], str(int(v))[4:6], str(int(v))[6:])
        for v in values
    )


def pick_varied(root, meta, folds, n, pool, rng):
    ids = list(meta[meta["Fold"].isin(folds)].index)
    rng.shuffle(ids)
    scored = []
    for pid in ids[: min(pool, len(ids))]:
        target = np.load(
            os.path.join(root, "ANNOTATIONS", "TARGET_{}.npy".format(pid))
        )[0]
        scored.append((len([c for c in np.unique(target) if c != VOID]), int(pid)))
    scored.sort(reverse=True)
    return [pid for _, pid in scored[:n]]


def export_one(root, meta, geo, pid, mean, std, out_dir, group):
    data = np.load(os.path.join(root, "DATA_S2", "S2_{}.npy".format(pid)))
    target = np.load(os.path.join(root, "ANNOTATIONS", "TARGET_{}.npy".format(pid)))[
        0
    ].astype(np.uint8)

    row = meta.loc[pid]
    dates = date_strings(row["dates-S2"])
    ref = np.datetime64(REF_DATE)
    offsets = np.array(
        [(np.datetime64(d) - ref) / np.timedelta64(1, "D") for d in dates],
        dtype=np.int64,
    )
    if len(offsets) != data.shape[0]:
        print(
            "  skipping {}: {} dates for {} images".format(
                pid, len(offsets), data.shape[0]
            )
        )
        return None

    bundle = {
        "data": data,
        "dates": offsets,
        "target": target,
        "mean": mean,
        "std": std,
        "patch_id": np.int64(pid),
        "fold": np.int64(row["Fold"]),
        "tile": str(row.get("TILE", "")),
        "ref_date": REF_DATE,
        "group": np.int64(group["id"]),
        "group_label": group["label"],
    }
    if geo is not None:
        shape = geo.loc[pid, "geometry"]
        minx, miny, maxx, maxy = shape.bounds
        bundle["bounds"] = np.array([miny, minx, maxy, maxx], dtype=np.float64)
        if hasattr(shape, "exterior"):
            ring = np.array(shape.exterior.coords)
            bundle["polygon"] = np.stack([ring[:, 1], ring[:, 0]], axis=1)

    path = os.path.join(out_dir, group["dir"], "patch_{}.npz".format(pid))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **bundle)
    size_mb = os.path.getsize(path) / 1e6
    n_classes = int(len([c for c in np.unique(target) if c != VOID]))

    print(
        "  {:<40} {:>2} dates  {:>2} crops  {:>5.1f} MB".format(
            "{}/patch_{}.npz".format(group["dir"], pid),
            data.shape[0],
            n_classes,
            size_mb,
        )
    )
    return {
        "file": "{}/patch_{}.npz".format(group["dir"], pid),
        "name": "patch_{}".format(pid),
        "label": "Patch {}".format(pid),
        "patch_id": int(pid),
        "fold": int(row["Fold"]),
        "tile": str(row.get("TILE", "")),
        "group": group["id"],
        "group_label": group["label"],
        "n_dates": int(data.shape[0]),
        "n_classes": n_classes,
        "size_mb": round(size_mb, 1),
        "first_date": dates[0],
        "last_date": dates[-1],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_folder", required=True)
    p.add_argument("--fold", default=1, type=int, help="Which trained fold this is for")
    p.add_argument("--n", default=10, type=int, help="Patches per group")
    p.add_argument("--out", default="./samples")
    p.add_argument("--seed", default=7, type=int)
    p.add_argument(
        "--pool",
        default=120,
        type=int,
        help="How many candidates to inspect per group before picking the most varied",
    )
    p.add_argument("--zip", dest="make_zip", action="store_true", default=True)
    p.add_argument("--no-zip", dest="make_zip", action="store_false")
    args = p.parse_args()

    root = args.dataset_folder
    train_folds, val_fold, test_fold = FOLD_SEQUENCE[args.fold - 1]
    print(
        "Fold {}: trained on {}, validated on {}, tested on {}".format(
            args.fold, train_folds, val_fold, test_fold
        )
    )

    meta = gpd.read_file(os.path.join(root, "metadata.geojson"))
    meta.index = meta["ID_PATCH"].astype(int)
    meta = meta.sort_index()
    geo = to_wgs84(meta)

    plan = [(GROUPS[0], train_folds), (GROUPS[1], test_fold)]
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for group, folds in plan:
        mean, std = load_norm(root, folds)
        chosen = pick_varied(root, meta, folds, args.n, args.pool, rng)
        print("\n{} -- folds {} -- patches {}".format(group["label"], folds, chosen))
        for pid in chosen:
            entry = export_one(root, meta, geo, pid, mean, std, args.out, group)
            if entry:
                manifest.append(entry)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(
            {
                "fold": args.fold,
                "train_folds": train_folds,
                "val_fold": val_fold,
                "test_fold": test_fold,
                "groups": [dict(g) for g, _ in plan],
                "samples": manifest,
            },
            fh,
            indent=2,
        )

    shutil.copy(
        os.path.join(root, "NORM_S2_patch.json"),
        os.path.join(args.out, "NORM_S2_patch.json"),
    )

    total = sum(m["size_mb"] for m in manifest)
    print("\n{} bundles, {:.1f} MB total, in {}".format(len(manifest), total, args.out))

    if args.make_zip:
        archive = shutil.make_archive(args.out.rstrip("/"), "zip", args.out)
        print(
            "Zipped to {} ({:.1f} MB)".format(archive, os.path.getsize(archive) / 1e6)
        )


if __name__ == "__main__":
    main()

"""The web server behind the viewer.

Start it with:

    uv run python webapp/server.py --res_dir results --fold 1

It serves the single page in webapp/static/ and exposes a small JSON API that
the page calls to list samples, run the model, and fetch the rendered maps.
"""

import argparse
import io
import json
import os
import sys
import uuid
from collections import OrderedDict

import numpy as np
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from webapp import inference  # noqa: E402
from webapp.pastis_meta import (  # noqa: E402
    BANDS,
    CLASS_COLORS,
    CLASS_NAMES,
    PATCH_AREA_HA,
    PIXEL_SIZE_M,
    VOID_CLASS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

app = FastAPI(title="U-TAE crop segmentation viewer", docs_url=None, redoc_url=None)

# Filled in by main(); kept in a dict so the route handlers can see the values
# without a pile of globals.
STATE = {
    "bundle": None,
    "error": None,
    "samples_dir": os.path.join(HERE, "samples"),
    "fallback_norm": None,
}

# The rendered PNGs for the last few predictions. The browser asks for them one
# request after the JSON, so they have to survive in between -- but there is no
# reason to keep every prediction ever made.
CACHE = OrderedDict()
CACHE_LIMIT = 6


def _remember(token, maps):
    CACHE[token] = maps
    while len(CACHE) > CACHE_LIMIT:
        CACHE.popitem(last=False)


def _load_fallback_norm(samples_dir):
    """NORM_S2_patch.json, if the export dropped a copy next to the samples.

    Only used for files that arrive without their own mean/std baked in.
    """
    path = os.path.join(samples_dir, "NORM_S2_patch.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            vals = json.load(fh)
        means = np.stack([vals["Fold_{}".format(f)]["mean"] for f in range(1, 6)])
        stds = np.stack([vals["Fold_{}".format(f)]["std"] for f in range(1, 6)])
        return means.mean(axis=0), stds.mean(axis=0)
    except (KeyError, ValueError):
        return None


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #


@app.get("/api/status")
def status():
    bundle = STATE["bundle"]
    payload = {
        "model_loaded": bundle is not None,
        "error": STATE["error"],
        "classes": [
            {"id": i, "name": n, "color": c, "void": i == VOID_CLASS}
            for i, (n, c) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
        ],
        "bands": BANDS,
        "patch": {
            "size_px": 128,
            "pixel_size_m": PIXEL_SIZE_M,
            "area_ha": PATCH_AREA_HA,
        },
    }
    if bundle is not None:
        payload["run"] = bundle.meta
        payload["config"] = {
            k: bundle.config.get(k)
            for k in (
                "model",
                "encoder_widths",
                "decoder_widths",
                "n_head",
                "d_model",
                "d_k",
                "agg_mode",
                "num_classes",
                "epochs",
                "batch_size",
                "lr",
                "ref_date",
            )
        }
    return payload


def _resolve_sample(name):
    """Turn a manifest entry into an absolute path, refusing to leave samples/.

    Entries look like "group1/patch_10005.npz", so plain basename stripping is
    not enough -- but a path that climbs out with ".." must still be rejected.
    """
    directory = STATE["samples_dir"]
    full = os.path.normpath(os.path.join(directory, name))
    if os.path.commonpath([os.path.abspath(full), directory]) != directory:
        raise HTTPException(status_code=400, detail="Bad sample path.")
    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="No sample named {}".format(name))
    return full


def _read_manifest():
    path = os.path.join(STATE["samples_dir"], "manifest.json")
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh), path


@app.get("/api/samples")
def samples():
    directory = STATE["samples_dir"]
    manifest, manifest_path = _read_manifest()

    if manifest:
        entries = manifest.get("samples", manifest)
        available = [
            item
            for item in entries
            if item.get("file") and os.path.exists(os.path.join(directory, item["file"]))
        ]
        if available:
            return {
                "samples": available,
                "groups": manifest.get("groups", []),
                "folds": {
                    "train": manifest.get("train_folds"),
                    "val": manifest.get("val_fold"),
                    "test": manifest.get("test_fold"),
                },
                "source": manifest_path,
            }

    # No manifest: fall back to whatever .npz files are lying about, including
    # inside subfolders, so a hand-assembled samples/ directory still works.
    if not os.path.isdir(directory):
        return {"samples": [], "groups": [], "source": None}
    found = []
    for base, _, files in os.walk(directory):
        for fname in sorted(files):
            if not fname.endswith(".npz"):
                continue
            full = os.path.join(base, fname)
            rel = os.path.relpath(full, directory)
            found.append(
                {
                    "file": rel,
                    "name": os.path.splitext(fname)[0],
                    "label": os.path.splitext(fname)[0],
                    "size_mb": round(os.path.getsize(full) / 1e6, 1),
                }
            )
    return {"samples": found, "groups": [], "source": directory}


def _require_model():
    if STATE["bundle"] is None:
        raise HTTPException(
            status_code=503,
            detail=STATE["error"] or "No model is loaded.",
        )
    return STATE["bundle"]


@app.post("/api/predict")
def predict(sample: str = Form(default="")):
    bundle = _require_model()

    # Form(...) would answer a missing field with a 422 validation dump; the page
    # only ever shows `detail`, so a plain 400 reads better.
    if not sample.strip():
        raise HTTPException(status_code=400, detail="Send a sample name.")

    path = _resolve_sample(sample)
    name = os.path.splitext(os.path.basename(path))[0]

    try:
        loaded = inference.load_sample(
            path, name=name, fallback_norm=STATE["fallback_norm"]
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload, maps = inference.predict(bundle, loaded)
    token = uuid.uuid4().hex[:16]
    _remember(token, maps)
    payload["token"] = token
    payload["layers"] = sorted(maps.keys())
    return JSONResponse(payload)


@app.get("/api/map/{token}/{which}.png")
def get_map(token: str, which: str):
    maps = CACHE.get(token)
    if maps is None:
        raise HTTPException(status_code=404, detail="This prediction has expired. Run it again.")
    blob = maps.get(which)
    if blob is None:
        raise HTTPException(status_code=404, detail="No layer named {}".format(which))
    return Response(content=blob, media_type="image/png")


# --------------------------------------------------------------------------- #
#  Static page
# --------------------------------------------------------------------------- #


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res_dir", default="results", help="Folder holding conf.json and Fold_N/")
    p.add_argument("--fold", default=1, type=int)
    p.add_argument("--weights", default=None, help="Override the checkpoint path")
    p.add_argument("--samples", default=os.path.join(HERE, "samples"))
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", default=8000, type=int)
    args = p.parse_args()

    STATE["samples_dir"] = os.path.abspath(args.samples)
    STATE["fallback_norm"] = _load_fallback_norm(STATE["samples_dir"])

    try:
        STATE["bundle"] = inference.load_bundle(
            args.res_dir, args.fold, device=args.device, weights=args.weights
        )
        meta = STATE["bundle"].meta
        print(
            "Loaded fold {} on {} -- {:,} parameters".format(
                meta["fold"], meta["device"], meta["n_params"]
            )
        )
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        # Not fatal. The page still opens and explains what is missing, which is
        # more useful than a stack trace in the terminal.
        STATE["error"] = str(exc)
        print("Model not loaded: {}".format(exc))

    print("Samples from {}".format(STATE["samples_dir"]))
    print("Open http://{}:{}".format(args.host, args.port))

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

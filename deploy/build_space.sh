#!/usr/bin/env bash
#
# Assembles the Hugging Face Space repo in a staging directory.
#
#   bash deploy/build_space.sh [output_dir]
#
# This repo cannot be pushed to a Space directly: .gitignore excludes results/,
# /samples/, *.pth.tar, *.pkl and *.npz, so a plain push would ship a Space with
# no model and no patches. The staging directory is a separate, self-contained
# copy holding only what the server reads at runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/deploy/space-build}"

die() { echo "build_space: $*" >&2; exit 1; }

[ -f "$ROOT/samples/manifest.json" ] ||
  die "samples/manifest.json is missing. Run notebooks/export_samples.py on the
     GPU box and unpack its output into samples/."
[ -f "$ROOT/results/Fold_1/model.pth.tar" ] ||
  die "results/Fold_1/model.pth.tar is missing. Download the trained run into
     results/ before building."

rm -rf "$OUT"
mkdir -p "$OUT/webapp" "$OUT/samples" "$OUT/results/Fold_1"

# The whole package: model_utils imports src.backbones.utae, which pulls the
# rest of src/ in with it.
cp -r "$ROOT/src" "$OUT/src"

for f in __init__.py inference.py pastis_meta.py server.py; do
  cp "$ROOT/webapp/$f" "$OUT/webapp/$f"
done
cp -r "$ROOT/webapp/static" "$OUT/webapp/static"

# group2/ is left behind: 111 MB the frontend never shows.
cp "$ROOT/samples/manifest.json" "$OUT/samples/manifest.json"
cp "$ROOT/samples/NORM_S2_patch.json" "$OUT/samples/NORM_S2_patch.json"
cp -r "$ROOT/samples/group1" "$OUT/samples/group1"

# last.pth.tar is only for --resume and conf_mat.pkl is never read by the viewer.
cp "$ROOT/results/conf.json" "$OUT/results/conf.json"
for f in model.pth.tar test_metrics.json trainlog.json; do
  cp "$ROOT/results/Fold_1/$f" "$OUT/results/Fold_1/$f"
done

cp "$ROOT/deploy/Dockerfile" "$OUT/Dockerfile"
cp "$ROOT/deploy/requirements.txt" "$OUT/requirements.txt"
cp "$ROOT/deploy/.gitattributes" "$OUT/.gitattributes"
cp "$ROOT/deploy/space_README.md" "$OUT/README.md"

find "$OUT" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$OUT" -name '*.py[cod]' -delete

# Trim the manifest to the bundles that were actually copied. Field names stay
# exactly as export_samples.py wrote them -- the frontend reads them by name.
python3 - "$OUT/samples/manifest.json" <<'PY'
import json
import os
import sys

path = sys.argv[1]
root = os.path.dirname(path)
with open(path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)

manifest["samples"] = [
    s
    for s in manifest.get("samples", [])
    if s.get("group") == 1 and os.path.exists(os.path.join(root, s["file"]))
]
manifest["groups"] = [g for g in manifest.get("groups", []) if g.get("id") == 1]

with open(path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)

print("manifest: {} bundle(s)".format(len(manifest["samples"])))
PY

echo "staged $(du -sh "$OUT" | cut -f1) in $OUT"

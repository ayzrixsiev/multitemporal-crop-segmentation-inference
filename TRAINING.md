# Training U-TAE on PASTIS with a free GPU

The model is small (1.1M parameters). The dataset is not: PASTIS is **29 GB** and one
epoch reads roughly 17 GB off disk, so **I/O is the bottleneck, not the GPU**. Every
decision below follows from that.

---

## Recommended: Kaggle Notebooks

Free tier gives 30 GPU-hours/week on a P100 (16 GB) or 2×T4, sessions up to 12 h,
20 GB of persistent output, and — the part that matters — **datasets mounted
read-only at `/kaggle/input`, which cost no session time and never need
re-downloading.**

### One-time: get PASTIS into Kaggle

Kaggle allows 200 GB per dataset, so 29 GB is comfortable. Download once from Zenodo
([10.5281/zenodo.5012942](https://doi.org/10.5281/zenodo.5012942)), then push it up:

```bash
pip install kaggle                      # put your kaggle.json in ~/.kaggle/
unzip PASTIS.zip -d pastis_upload/      # -> pastis_upload/PASTIS/

kaggle datasets init -p pastis_upload/
# edit pastis_upload/dataset-metadata.json: set "title" and "id" (<username>/pastis)
kaggle datasets create -p pastis_upload/ --dir-mode zip
```

This is the slow step — one 29 GB download plus one 29 GB upload. It only happens
once; after that every notebook session mounts it instantly.

### One-time: push this repo

Kaggle clones from GitHub, so anything still sitting uncommitted on your machine will
not be there:

```bash
git add -A && git commit -m "semantic track ready to train" && git push
```

### Every session: the notebook

Create a notebook, then in the sidebar set **Accelerator → GPU P100** and
**Internet → On**, and add your PASTIS dataset under **Input**.

```python
!git clone https://github.com/ayzrixsiev/utae-crop-temporal-segmentation.git /kaggle/working/utae
%cd /kaggle/working/utae
!pip install -q geopandas
```

```python
!python train_semantic.py \
    --dataset_folder /kaggle/input/pastis/PASTIS \
    --res_dir /kaggle/working/results \
    --fold 1 \
    --epochs 60 \
    --batch_size 4 \
    --num_workers 4 \
    --device cuda
```

Then run it with **Save Version → Save & Run All (Commit)**, not interactively.
Committed runs execute in the background for the full 12 h and survive closing the
tab; interactive sessions die when you go idle.

Everything under `/kaggle/working` is kept as the notebook's output, so
`results/Fold_1/model.pth.tar` (about 4 MB) and the JSON logs come back with you.

### Then: the report

```python
!python scripts/make_report.py \
    --res_dir /kaggle/working/results --fold 1 \
    --dataset_folder /kaggle/input/pastis/PASTIS \
    --out /kaggle/working/report/index.html
```

Download `report/index.html` — it is fully self-contained, every image inlined.

### Sizing the run to the 12-hour cap

`train_semantic.py` has **no resume**: it trains, then tests, then exits. If the
session is killed mid-run you lose everything except the last saved best-mIoU
checkpoint. So pick an epoch count that fits in one session.

Budget roughly 4–8 min/epoch on a P100 at `--batch_size 4 --num_workers 4`. Time a
short run first, then scale. `--epochs 60` on a single fold is the safe starting
point and lands within about a point of the paper's 100-epoch number. Run
`--fold 1` only; the five-fold cross-validation is five separate runs and the
`overall.json` summary only appears when all five exist.

---

## The alternatives, ranked

**Google Colab (free).** T4, similar speed, but no persistent disk — free Drive is
15 GB, which PASTIS does not fit in, and Drive is slow for many small files.
Disconnects more aggressively than Kaggle. Only better than Kaggle if you already pay
for Drive storage.

**Lightning AI Studios (free tier).** Around 22 free GPU-hours/month, but with a
**persistent filesystem**, so you `wget` PASTIS straight from Zenodo once and it stays
there. Fewer hours than Kaggle, less setup friction. Good second choice.

**SageMaker Studio Lab (free).** 15 GB persistent, 4-hour GPU sessions. The storage is
too small for PASTIS.

**RunPod / Vast.ai (paid, roughly $0.20–0.35/h).** Not free, but worth knowing: an
RTX 3090 or 4090 with fast NVMe and a fast link pulls PASTIS from Zenodo in minutes,
and a full 100-epoch single-fold run costs about $3 total with no session cap and no
upload dance. If the Kaggle upload turns into a multi-day fight, this is the escape
hatch.

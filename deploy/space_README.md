---
title: CropScope
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# CropScope

Crop-type segmentation of Sentinel-2 satellite image time series with U-TAE
(Sainte Fare Garnot et al.), reimplemented from
[VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps) and trained on the
[PASTIS](https://github.com/VSainteuf/pastis-benchmark) dataset.

The unit of input is **one year, not one image**. A patch is 128 × 128 pixels at
10 m per pixel (163.84 ha) observed on 33–61 Sentinel-2 acquisition dates, each
with 10 spectral bands. The model reads the whole stack at once and returns one
crop label per pixel across 20 classes.

This Space runs a single checkpoint: fold 1 of the five-fold PASTIS split, 65
epochs, 1,087,260 parameters, semantic segmentation only. On the held-out test
fold it scores **82.98 % overall accuracy** and **59.64 mIoU**
(`results/Fold_1/test_metrics.json`).

Ten patches are bundled with the app; pick one on the left to run it. Inference
happens on CPU and takes a few seconds per patch.

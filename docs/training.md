# RiverScope training and evaluation pipeline

This guide defines the reproducible baseline pipeline for binary water
segmentation on the active RiverScope / PlanetScope dataset. It uses the
published geographic `train`, `valid`, and `test` partitions unchanged.

## Scope and split lock

`data/processed/riverscope/geometry-v2/manifests/` is the source of truth for
the derived arrays. Every training sample includes a four-band PlanetScope
image (Blue, Green, Red, NIR), binary water target, validity mask, thick
8-neighbour shoreline, and squared shoreline distance map (`d²`).

- Train is the only source for normalization statistics and parameter fitting.
- Validation controls checkpoints, early stopping, and all configuration
  selection.
- Test is intentionally not accepted by the training or grid-search entry
  points. It is loaded only in the final evaluation notebook after a
  configuration and checkpoint have been frozen.
- The loader checks that published tile paths and `reach_id` geographic groups
  do not overlap across the splits it is asked to build.

The geometry pipeline has already excluded published all-zero images according
to the versioned quality manifest. Source labels equal to `3` are represented
by `valid_mask == False`; they are not a third class and never affect a loss,
metric, predicted-boundary comparison, or error visualization.

## Data loading, normalization, and padding

`RiverScopeSegmentationDataset` reads an immutable published image and its
aligned `.npy` geometry maps. It accepts either the original ZIP or an
extracted RiverScope directory and opens the ZIP once per worker process.
`fit_channel_statistics` accepts only `manifests/train.csv`; it calculates
per-channel mean and standard deviation over valid train pixels and saves
those numbers in every run's artifacts. Validation and test use the saved
statistics without recomputation.

The initial audit shows normalized PlanetScope sample values in the `[0, 1]`
range. Standardization still uses train-only statistics because channel
distributions differ substantially, especially NIR. No statistic is inferred
from validation or test data.

The U-Net internally pads a 500×500 input to the next multiple of its depth
(512×512 at depth four) and crops its logits to the original size. Targets,
validity masks, `d²`, losses, and metrics therefore remain at 500×500 and do
not contain artificial padded pixels. Synthetic tests cover this round trip.

## Losses

The baseline has two logits per pixel and uses Cross Entropy without a manual
softmax:

```text
L = sum(valid * CE(logits, target)) / sum(valid)
```

The comparable shoreline experiment changes only the loss:

```text
w = 1 + alpha * exp(-d² / (2 * sigma²))
L = sum(valid * w * CE(logits, target)) / sum(valid)
```

`d²` comes directly from `geometry-v2` and is never regenerated for a choice
of `alpha` or `sigma`. The denominator is valid-pixel count, not sum of
weights. A completely invalid batch returns no loss and is skipped safely.
Class weighting and augmentation are not part of these baseline experiments.

## Metrics and degenerate shoreline cases

Regional metrics are masked water Dice, water IoU, and masked pixel accuracy.
They are reported globally by summing valid-pixel confusion counts and per
tile for diagnosis. Accuracy is secondary because class imbalance can make it
look favorable while water quality is poor.

Shoreline metrics use the geometry-v2 thick 8-neighbour convention:

- Boundary F1 has a fixed tolerance of two pixels.
- ASSD and HD95 use symmetric Euclidean pixel distances between predicted and
  reference shoreline pixels.
- These metrics are calculated per tile, then grouped by `reach_id`. The
  selection metric is the macro mean of group-level `selection_assd_px`, so a
  large geographic group cannot dominate the decision.

Degenerate outcomes are kept visible in every per-tile table and group
summary:

| Case | Boundary F1 | Standard ASSD / HD95 | Selection ASSD |
| --- | --- | --- | --- |
| Both boundaries empty | 1 | 0 | 0 |
| Only predicted boundary empty | 0 | undefined (`null`) | tile diagonal penalty |
| Only reference boundary empty | 0 | undefined (`null`) | tile diagonal penalty |
| Both boundaries present | standard result | standard result | ASSD |

The finite diagonal penalty prevents an undetected shoreline from disappearing
when group metrics are averaged. Reports also include counts of every case;
the standard ASSD and HD95 means include only defined values and are labeled
accordingly.

## Checkpoints, scheduling, and reproducibility

The checkpoint selector prioritizes lower geographic-macro selection ASSD,
but only within a validation-Dice safety band. An epoch more than `0.01` Dice
below the current selected checkpoint cannot replace it. An epoch more than
`0.01` Dice above it becomes the new checkpoint even when ASSD is worse,
protecting regional segmentation from degradation. Within the band, ASSD
must improve by at least `0.01` pixels. Early stopping is always active with
patience 20 and restores that best checkpoint. `ReduceLROnPlateau` monitors
the same selection ASSD with an absolute threshold of `0.01` pixels and
default patience 8.

Every run records the full configuration, seed, package versions, Git commit,
chosen device, deterministic-mode setting, train-only channel statistics,
metrics per epoch, best checkpoint, and a final completed or failed status.
The seed function covers Python, NumPy, PyTorch CPU/CUDA, DataLoader generator,
and worker initialization. Deterministic mode disables `cudnn.benchmark`.
This improves repeatability on a fixed stack but cannot guarantee bit-for-bit
identity across different hardware, CUDA, drivers, or PyTorch versions.

## Notebook workflow

1. `03_grid_search.ipynb` runs a progressive search using train and validation
   only. It first evaluates 12 CE trials (LR × base channels × batch size),
   then 8 weighted-CE trials with the selected architecture and batch size
   fixed. The two finalists are each repeated with seeds 42, 123, and 2026,
   for 26 train/validation runs in total. Every terminal trial is persisted;
   completed configuration hashes are skipped when the notebook is rerun.
2. `04_train_model.ipynb` accepts a manual frozen configuration or the winner
   persisted by the search. It runs the one- or two-tile overfit smoke test
   before a full train/validation run, renders train/validation curves, and
   loads the saved best validation checkpoint.
3. `05_test_visualization.ipynb` accepts only that frozen checkpoint and its
   saved train statistics. It does no model selection. It exports tile metrics
   and representative test visualizations that clearly mark invalid pixels
   excluded from all calculations.

All generated outputs live under `artifacts/`, which is ignored by Git. Run
CPU tests and smoke tests locally; use a CUDA-equipped environment for the
full grid search.

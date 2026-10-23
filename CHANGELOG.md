# Changelog

All notable changes to this repository are documented here.

## Unreleased

- Add explicit, pinned CPU and CUDA 12.8 PyTorch runtime profiles. The CUDA
  profile supports RTX 50-series GPUs and is now required by the documented
  full-training and grid-search setup.

- Added a reproducible RiverScope train-validation-test pipeline with
  manifest-backed inputs, train-only valid-pixel normalization, deterministic
  seed controls, CPU/CUDA provenance, and internal U-Net padding that never
  exposes artificial pixels to losses or metrics.
- Added masked CE and the documented unnormalized weighted CE, regional and
  grouped shoreline metrics, explicit degenerate-boundary reporting, ASSD with
  a Dice safety constraint for checkpointing, mandatory early stopping, and
  persistent experiment/checkpoint/trial artifacts.
- Added progressive, resumable CE and weighted-CE grid-search helpers plus
  numbered grid-search, training, and test-visualization notebooks.
- Added synthetic training tests for validity masks, padded model output,
  weighted-loss normalization, shoreline edge cases, split leakage, and
  checkpoint selection.
- Documented the training, selection, metric, reproducibility, and test-lock
  conventions.
- Treat the 157 observed, undocumented RiverScope label-`3` pixels as explicit
  no-data in derived geometry. They are retained in raw labels and excluded
  from shorelines, losses, and metrics, with per-tile counts in manifests.
- Suppress the repeated malformed `GDAL_NODATA` metadata warning emitted by
  `tifffile` while reading RiverScope labels.
- Added a versioned RiverScope geometry-processing pipeline that derives
  binary water masks, explicit validity masks, thick 8-neighbour shorelines,
  and reusable float32 squared-distance maps without modifying raw data.
- Added a per-split processing CLI, reproducible derived-data manifests, and
  documentation of the shoreline, validity, cache, and overwrite conventions.
- Added tests for diagonal water/land contacts, invalid pixels, outer-tile
  boundaries, uniform masks, label policy, and distance-map representation.
- Added a configurable PyTorch U-Net for four-channel 500x500 PlanetScope
  tiles. It pads inputs to 512x512, uses transposed convolutions in the
  decoder, returns cropped logits for Cross-Entropy training, and provides a
  separate inference-only probability interface.
- Refactored the U-Net into reusable encoder and decoder blocks, independent
  padding and prediction utilities, and dedicated unit and integration tests.
- Added an explicit, MD5-verified downloader for the official RiverScope ZIP.
- Added support for reading RiverScope split CSVs and extracting reproducible
  PlanetScope image/label samples from either a ZIP or an extracted folder.
- Reworked the exploration notebook around RiverScope / PlanetScope quality
  control: metadata, alignment, original and binary labels, RGB/NIR views,
  coast zooms, distance and weight maps, class balance, split audits, and a
  manual review table.
- Documented faithful acquisition, data handling, attribution, and the
  separation between this repository's MIT code license and external data
  terms.
- Declared the notebook runtime as an installable optional dependency set and
  documented use of the repository's existing `.venv` environment.
- Rejected unsafe tile identifiers before materializing sample files so output
  paths cannot escape the selected destination.
- Added a versioned RiverScope quality-exclusion manifest and reusable
  filtering utilities for derived samples, training, and evaluation workflows.
- Materialize selected tiles with unique names derived from their published
  normalized paths, preventing collisions between crops from one scene.
- Expanded the quality manifest to the 98 exact tiles identified by the
  all-zero-image audit across the published splits.
- Recorded the audited `nonzero_values` count for each excluded tile.
- Defined the numerical all-zero-image criterion and documented where the
  versioned exclusion manifest is applied.

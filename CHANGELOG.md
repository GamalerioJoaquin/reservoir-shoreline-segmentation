# Changelog

All notable changes to this repository are documented here.

## Unreleased

- Added a configurable PyTorch U-Net for four-channel 500x500 PlanetScope
  tiles. It pads inputs to 512x512, uses transposed convolutions in the
  decoder, returns cropped logits for Cross-Entropy training, and provides a
  separate inference-only probability interface.
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

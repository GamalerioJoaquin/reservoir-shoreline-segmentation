# Changelog

All notable changes to this repository are documented here.

## Unreleased

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

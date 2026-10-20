# Reservoir Shoreline Segmentation

Exploration and reproducible quality control for water segmentation and
shoreline analysis using the public RiverScope / PlanetScope dataset.

## What is included

- `notebooks/01_dataset_exploration.ipynb` loads official RiverScope split CSVs,
  materializes a reproducible sample, checks GeoTIFF metadata and alignment,
  visualizes RGB/NIR compositions and labels, inspects coast boundaries and
  distance-based weights, and audits split identifiers.
- `src/reservoir_segmentation/` contains reusable utilities for downloading
  and verifying the official archive, reading split CSVs, extracting samples,
  reading TIFF metadata, inspecting boundaries, and computing distance maps.
- `tests/` covers the reusable utilities with synthetic inputs.

The notebook preserves the published multiclase labels: `0 = background`,
`1 = river water`, and `2 = other water`. Its binary inspection view treats
labels `1` and `2` as water, but does not overwrite the original mask.

The notebook reports observations about data quality. It does not claim model
accuracy and does not silently repair annotations or interpret nodata as land.

## Obtain RiverScope faithfully

The authoritative download is the [RiverScope Zenodo record](https://zenodo.org/records/15376394).
The official archive is approximately 8.1 GB compressed and contains the
published CSVs and co-registered products. The official repository documents
the layout and provides the same download link at
<https://github.com/cvl-umass/riverscope>.

For the least transformed copy:

1. Download `RiverScope.zip` from Zenodo, or let the notebook call the explicit
   downloader by setting `DOWNLOAD_DATASET = True`.
2. Verify MD5 `910f3ffc046cbf60df30c27937555fa5`.
3. Keep the archive unchanged and extract it without converting, resampling,
   renaming, or recompressing the GeoTIFFs.
4. Point `RIVERSCOPE_DATA` to the extracted folder, or place the archive at
   `data/raw/RiverScope.zip`.

The downloader validates the MD5 before replacing the final file and extracts
only the selected sample when the notebook reads a ZIP. The complete archive
and generated data stay outside Git.

## Git Bash setup on Windows

The repository has one environment for all development work. Changes in scope
do not require a new environment.

From Git Bash:

~~~
cd /c/Users/joaco/Projects/reservoir-shoreline-segmentation
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,dev]"
python -m ipykernel install --user --name reservoir-shoreline-segmentation --display-name "Python (Reservoir Shoreline Segmentation)"
~~~

Open the notebook with:

~~~
.venv/Scripts/jupyter.exe notebook notebooks/01_dataset_exploration.ipynb
~~~

If activation is unavailable, use the environment directly:

~~~
.venv/Scripts/python.exe -m pip install -e ".[notebooks,dev]"
.venv/Scripts/jupyter.exe notebook notebooks/01_dataset_exploration.ipynb
~~~

PowerShell users can activate the same environment with
.venv/Scripts/Activate.ps1; this is only a shell-specific alternative.

## Dataset location

The notebook first checks `RIVERSCOPE_DATA`, then `data/raw/riverscope`, and
finally `data/raw/RiverScope.zip`. Keep downloaded data outside Git; the
repository ignores `data/raw/`, `data/processed/`, and archive files.

~~~
export RIVERSCOPE_DATA=/d/datasets/riverscope
jupyter notebook notebooks/01_dataset_exploration.ipynb
~~~

PowerShell equivalent:

~~~powershell
$env:RIVERSCOPE_DATA = "C:\datasets\riverscope"
~~~

## Run tests and notebook

~~~
python -m pytest
python -m ruff check src tests
jupyter notebook notebooks/01_dataset_exploration.ipynb
~~~

If no dataset is available, the notebook still opens and prints the official
download URL and expected layout.

## License

Original code in this repository is MIT-licensed; see `LICENSE`. RiverScope's
official repository is marked CC0-1.0, while the Zenodo archive and the
underlying Planet imagery must be used according to their own current terms.
See [docs/licensing.md](docs/licensing.md) before sharing data or derived
products.

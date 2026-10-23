# RiverScope geometry processing

`scripts/prepare_riverscope.py` converts labels from the official RiverScope
ZIP or an extracted dataset into derived, versioned arrays. It never modifies
source GeoTIFFs, the archive, or the published split CSVs.

Run it from the repository root after obtaining the official archive:

```powershell
.venv/Scripts/python.exe scripts/prepare_riverscope.py data/raw/RiverScope.zip `
  --output data/processed/riverscope/geometry-v2
```

The default run processes `train`, `valid`, and `test`, preserves their
published assignments, and applies
[`configs/riverscope_quality_exclusions.csv`](../configs/riverscope_quality_exclusions.csv).
Use `--splits train valid` to derive selected splits, or `--overwrite` only
when intentionally regenerating existing output.

Git Bash uses forward slashes and backslash line continuations:

```bash
./.venv/Scripts/python.exe scripts/prepare_riverscope.py data/raw/RiverScope.zip \
  --output data/processed/riverscope/geometry-v2
```

`notebooks/02_distance_maps_qc.ipynb` accepts either the same ZIP archive or
an extracted RiverScope directory as its source, so the quality-control path
does not require a second copy of the data.

## Derived representation

For every retained tile, the output root contains:

- `binary_masks/<split>/<tile>.npy`: `bool`, where RiverScope labels 1 and 2
  become water and 0 becomes non-water;
- `valid_masks/<split>/<tile>.npy`: `bool` validity map;
- `boundaries/<split>/<tile>.npy`: a thick shoreline map;
- `distance_squared/<split>/<tile>.npy`: squared Euclidean distance (`d²`) in
  pixels, stored as `float32`;
- `manifests/<split>.csv`: published metadata plus paths to each derived map;
- `geometry_metadata.json`: the exact geometric and validity conventions.

The multiclass source label remains only in `data/raw` and is not overwritten.

The official documentation lists labels 0, 1, and 2, but the supplied archive
also contains 157 pixels labeled 3 across 23 retained tiles. This project
treats label 3 as undocumented no-data: it is retained unchanged in the raw
label, recorded in each derived manifest, set to `False` in `valid_masks`, and
excluded from shorelines, loss, and metrics.

## Coastline convention

The processor compares each pixel with its horizontal, vertical, and diagonal
neighbours. A pair defines coast only when both values are valid and one is
water while the other is non-water. Both endpoints are marked, yielding the
documented thick 8-neighbour boundary. The outer image frame, and transitions
to invalid data, are never treated as coast.

`d²` is calculated on the original 500 × 500 label tile before any future
padding or crops. A tile with no valid water/land transition receives `inf`,
so it does not acquire a fictional coastline at its edge.

## Validity policy

RiverScope does not currently provide an independent validity mask. The first
version therefore marks a pixel valid only when its label is finite, is not the
documented project no-data value 3, and is not a value passed explicitly with
`--label-nodata VALUE`. It does not infer cloud, shadow, or imagery nodata.
When those rules are defined, they must be added as a new documented geometry
version rather than silently changing this cache.

The cached `d²` deliberately has no `sigma` or `alpha` dimension. Training can
calculate `1 + alpha * exp(-d² / (2 * sigma²))` for every experiment without
repeating the distance transform.

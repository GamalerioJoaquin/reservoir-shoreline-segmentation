# RiverScope data-quality policy

## Purpose

This policy records reproducible decisions about unusable inputs while
preserving the published RiverScope archive and its official split CSVs
unchanged. It applies only to derived samples, training manifests, and
evaluation manifests.

## Versioned exclusion manifest

[`configs/riverscope_quality_exclusions.csv`](../configs/riverscope_quality_exclusions.csv)
is the source of truth for explicit exclusions. Each entry records the exact
published `normalized_planetscope_path`, an objective reason, a decision,
scope, evidence, and the audited `nonzero_values` count. A `planetscope_id`
may describe more than one cropped tile, so it is not precise enough for a
tile-level exclusion.

The current manifest excludes 98 tiles with the `all_zero_imagery` decision:
71 from `train`, 16 from `valid`, and 11 from `test`. The automated audit found
that every image-band value is finite and zero. This criterion concerns the
imagery only; a label that is entirely background is not, by itself, a reason
for exclusion.

## Numerical zero-value criterion

The non-zero-value check reads the complete image array for each tile and
records three quantities:

- `total_values`: number of values across all PlanetScope bands and pixels;
- `finite_values`: number of those values that are finite; and
- `nonzero_values`: number of values that differ from zero.

The `all_zero_imagery` criterion is satisfied only when
`finite_values == total_values` and `nonzero_values == 0`. It is therefore a
numeric data condition, not a display-dependent judgement. The notebook shows
the matching tile paths and whether each path is already present in the
manifest; it does not modify the official source files.

## Rules

- Never delete, overwrite, or edit files in the official archive.
- Do not exclude a tile merely because its label contains no water.
- An `all_zero_imagery` exclusion requires every image-band pixel to be zero;
  a dark rendering alone is insufficient evidence.
- Apply an objective exclusion rule consistently to every split in which it
  occurs. Preserve the published geographic split assignment for all retained
  rows.
- Keep exclusions visible in reports and state their count when publishing
  training or evaluation results.
- Filter by the exact `normalized_planetscope_path` before materializing a
  derived sample, gallery page, training manifest, or evaluation manifest.

## Using the manifest

```python
from reservoir_segmentation.dataset import (
    filter_riverscope_rows,
    load_quality_exclusions,
    read_riverscope_split,
)

exclusions = load_quality_exclusions("configs/riverscope_quality_exclusions.csv")
official_rows = read_riverscope_split("data/raw/RiverScope.zip", "train")
training_rows = filter_riverscope_rows(official_rows, exclusions)
```

The notebook applies the same manifest when materializing its sample, so an
excluded tile is not selected for visual review or future derived workflows.
It also provides a numerical non-zero-value check that reports tiles whose
image-band values are entirely finite zeros and shows whether each one is in
the manifest.

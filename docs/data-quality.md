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
scope, and evidence. A `planetscope_id` may describe more than one cropped
tile, so it is not precise enough for a tile-level exclusion.

The current manifest excludes the `normalized_planetscope_path` for the tile
whose `planetscope_id` is `20240627_033130_05_2478_3B_AnalyticMS_SR_clip`, for
`all_zero_imagery`. Inspection found that every pixel in all four image bands
is zero, even though the GeoTIFF reports valid image pixels and no image nodata
value. Its label is also entirely background, but that label condition is not
the reason for the decision.

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
It also provides an all-zero-image audit that creates a candidate table before
any new manifest entry is accepted.

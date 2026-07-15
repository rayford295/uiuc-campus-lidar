# comparison/ — OSM 2019 vs LiDAR building comparison

Produced by [`../../src/vgi_comparison.py`](../../src/vgi_comparison.py). Matches
LiDAR-detected building footprints (the remote-sensing ground truth) against **OSM 2019
`building=*`** via IoU (threshold 0.3, EPSG:6350), and reports completeness (recall),
commission, a gridded completeness surface (the spatial-bias map), and an
`omissions.geojson` layer (LiDAR buildings absent from OSM — the correction targets).

```bash
python src/vgi_comparison.py data/osm_buildings_2019.geojson
```

## Result — OSM 2019, campus tile (2 × 2 km)

| completeness (count) | completeness (area) | median IoU | LiDAR-only | OSM-only | OSM commission |
|---|---|---|---|---|---|
| **58.3%** | **79.4%** | 0.78 | 547 | 331 | 29.5% |

Three findings (`comparison_map.png`):

1. **OSM maps the big buildings, misses the small ones.** Area completeness (79%) far
   exceeds count completeness (58%): the 547 omissions are dominated by small structures —
   garages, sheds, outbuildings.
2. **The spatial-bias gradient is real and visible even within one tile.** The 250 m
   completeness surface drops from ~1.0 over the institutional campus core to **< 0.3 on
   the eastern residential strip**, where whole blocks of houses are absent from OSM 2019.
   Urban-core vs residential mapping effort differs sharply — exactly the bias this
   project targets, before even extending to a rural gradient.
3. **OSM-only features (331) need per-case interpretation**: geometry mismatches on
   complex footprints (IoU < 0.3 despite overlap), structures below the LiDAR detection
   threshold, and mapping errors — not automatically OSM commission in the map-error sense.

## Per-pixel comparison (`pixel/`, `src/pixel_comparison.py`)

Rasterizes both layers to a 0.2 m grid (see below for why not 0.1 m) and records a
per-pixel agreement category (both / LiDAR-only / OSM-only / neither) →
`pixel_diff_0p2m.tif` (inspect any pixel in ArcGIS/QGIS), plus metrics and a zoom crop.

| res | building-pixel IoU | pixel OA | Cohen κ | LiDAR-only | OSM-only |
|-----|--------------------|----------|---------|------------|----------|
| 0.2 m | 0.698 | 0.924 | 0.774 | 234,830 m² | 70,931 m² |

`pixel_disagreement_0p2m.png` corroborates the instance-level story: outside the eastern
residential strip (solid red = entire missing houses), disagreement is mostly a thin
edge fringe plus scattered whole small structures.

## Pipeline validation (previous run, kept for the record)

Before the OSM run, the same pipeline was validated against an independent LiDAR-derived
building extraction (cross-method check): completeness 99.8% (count) / 100.0% (area),
median IoU 0.94, pixel IoU 0.961, κ 0.974, with disagreement confined to a ~37,000 m²
edge fringe. 0.1 m and 0.2 m grids gave identical metrics (both layers are ≥0.5 m polygon
products), so 0.2 m is the standard resolution. This confirms the differences reported
above are OSM effects, not pipeline artifacts. (Artifacts of that run live in git history,
commit `c0038f0` and earlier.)

## Next step

Scale beyond the campus tile: the statewide OSM 2019 extracts
([release `osm-il-2019`](https://github.com/rayford295/vgi-spatial-bias/releases/tag/osm-il-2019))
cover the full urban→rural gradient; the LiDAR side requires additional
`IL_8County_PlusChampaign_2019_B19` tiles along that gradient.

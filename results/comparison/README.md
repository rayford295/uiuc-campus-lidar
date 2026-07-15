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

## OSM 2019 vs OSM 2026 — temporal validation (`osm2026/`, `temporal/`)

Current OSM (fetched via Overpass, `src/fetch_osm_current.py` →
`data/osm_buildings_2026.geojson`) run against the same 2019 LiDAR truth
(`src/vgi_comparison.py data/osm_buildings_2026.geojson osm2026`), plus a per-building
gap-tracking analysis (`src/temporal_validation.py` → `temporal/`):

| OSM snapshot | buildings | completeness (count) | completeness (area) | LiDAR-only |
|---|---|---|---|---|
| 2019 | 1,121 | 58.3% | 79.4% | 547 |
| 2026 | 1,657 | **81.9%** | **91.8%** | 238 |

**64% of the 2019 gaps (352 / 547) have since been filled by the community**
(`temporal/temporal_evolution.png`). Those buildings existed in the 2019 LiDAR and were
mapped later — so the 2019 omissions were genuine community under-mapping, not
buildings that didn't exist yet. This validates the bias measurement and shows OSM
completeness is strongly time-dependent: the eastern residential strip, nearly empty in
2019, is now largely mapped.

Caveats for the 2026-vs-2019-LiDAR numbers: (a) the 195 still-unmapped buildings may
include structures demolished after the 2019 flight (unmappable today); (b) the higher
2026 commission (33.5%) includes post-2019 construction that the 2019 LiDAR cannot
confirm. Temporal alignment (2019 vs 2019) remains the defensible headline comparison.

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

## Roads: OSM vs NAIP paved surface (`roads/`, `src/road_comparison.py`)

LiDAR has no road class, so roads are evaluated against the **NAIP paved layer**
(impervious − buildings). Forward test: buffer each OSM way by a class-dependent width
(+1.5 m positional tolerance) and measure the paved fraction inside ("RS support",
threshold 0.5). Reverse test: fraction of paved area within any buffered OSM way.

| OSM snapshot | ways (in tile) | length | supported (length) | median support | paved explained |
|---|---|---|---|---|---|
| 2019 | 3,383 | 246 km | **91.1%** | 0.83 | 54.9% |
| 2026 | 5,402 | 267 km | 90.8% | 0.89 | 58.6% |

Findings (`roads_2019.png`, `roads_2026.png`):

1. **Roads were already well-mapped in 2019** — 91% of OSM way length has pavement
   evidence, unlike buildings (58%). Campus road/footway networks attract early mapping.
2. **2026 growth is micro-mapping, not new coverage**: +60% segments but only +8%
   length (crossings, sidewalk links, driveways split into short ways), median support
   rising 0.83 → 0.89 (better geometric alignment).
3. **Unexplained paved (~41–45%) is dominated by parking lots and plazas** — consistent
   with `paved` being a superset of roads; these are area features OSM maps sparsely as
   `amenity=parking`, not missing centrelines.
4. **Caveats**: optical NAIP cannot see pavement under tree canopy, so shaded residential
   streets/footways (eastern strip) read as "unsupported" — a false alarm LiDAR does not
   suffer from; and unpaved `path`/`track` ways legitimately lack pavement support.

### Road network evolution 2019 → 2026 (`../temporal/roads_evolution.png`, `src/road_evolution.py`)

Every 2026 way is classified as *existed in 2019* (≥50% of its length within 5 m of the
2019 network) or *added*; added ways are then checked against the **2019** NAIP pavement:

| added ways | added length | already paved in 2019 | no 2019 evidence | 2019 ways gone |
|---|---|---|---|---|
| 440 | 20.6 km | **16.5 km (80%)** — filled gaps | 4.2 km | 116 (4.8 km) |

**80% of the length added to OSM since 2019 was already paved in the 2019 imagery** —
the same gap-fill logic as buildings (64%): the network existed, the map lagged. Blue
additions concentrate in the NE residential quarter, the same under-mapped area the
building analysis flagged. The 4.2 km without 2019 evidence is post-2019 construction,
unpaved ways, or canopy-hidden pavement; the 116 vanished ways are deletions/redraws.

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

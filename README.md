# UIUC Campus LiDAR — Detection & Segmentation

End-to-end, reproducible LiDAR analysis of the University of Illinois Urbana-Champaign
campus: a merged 2 × 2 km USGS 3DEP QL1 point cloud (80.8 M points) processed into
**building instances**, **individual trees**, a **bare-earth terrain model**, and a
**deep-learning semantic segmentation** of the ASPRS classes.

The [notebook](UIUC_campus_LiDAR_pipeline.ipynb) is fully self-contained — it downloads
the point cloud from I-GUIDE storage and runs everything top to bottom.

## Results

| | |
|---|---|
| ![buildings](outputs/detection/buildings_detected.png) | ![trees](outputs/detection/trees_detected.png) |
| **1,312 building instances** (footprint + height) | **11,777 individual trees** (height + crown) |
| ![fulltile](outputs/segmentation/seg_fulltile.png) | ![confusion](outputs/segmentation/seg_confusion.png) |
| **DGCNN** wall-to-wall semantic segmentation (whole 2×2 km) | val **OA 0.930 · mIoU 0.768** |

### Semantic segmentation models

Two point-cloud networks predict the five ASPRS classes from geometry + features, both
with a **spatial** west-train / east-val split (no leakage). Height-above-ground is the
key engineered feature.

| Model | OA | mIoU | Ground | Low Veg | Med Veg | High Veg | Building |
|-------|----|------|--------|---------|---------|----------|----------|
| PointNet (baseline) | 0.913 | 0.707 | 0.96 | 0.39 | 0.57 | 0.85 | 0.78 |
| **DGCNN (EdgeConv)** | **0.930** | **0.768** | 0.95 | **0.49** | **0.70** | **0.88** | **0.82** |

DGCNN's dynamic k-NN graph gives each point **local geometric context**, sharpening
building edges and improving every vegetation class over the per-point PointNet.

## VGI comparison (in progress)

The end goal is **detecting & correcting spatial bias in VGI (OpenStreetMap) using remote
sensing**: LiDAR building footprints are the objective ground truth, matched against OSM
`building=*` to reveal where OSM under-maps. `src/vgi_comparison.py` implements the
IoU matching → completeness → gridded bias-map pipeline:

```bash
python src/vgi_comparison.py osm_buildings_2019.geojson   # 2019 = temporally matched to LiDAR
```

The committed run in `outputs/comparison/` is a **pipeline test against a LiDAR-derived
reference** (cross-method check, 99.8% agreement) — see that folder's README. The real
OSM comparison, over an urban→rural gradient where bias actually varies, is the next step.

**The 2019 OSM data is now in hand** (temporally matched to the 2019 LiDAR acquisition):
campus-extent subsets are committed at the repo root — `osm_buildings_2019.geojson`
(1,121 buildings vs 1,312 LiDAR-detected instances) and `osm_roads_2019.geojson`
(3,614 segments) — and the full Illinois statewide shapefiles (1.20 M buildings,
765 K roads) are archived as
[release `osm-il-2019`](https://github.com/rayford295/uiuc-campus-lidar/releases/tag/osm-il-2019)
for scaling the analysis across the urban→rural gradient. The broader research framing
(evaluating & calibrating VGI with multimodal remote sensing) is in
[`PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md).

## Data

- **Source:** USGS 3DEP Lidar Point Cloud, `IL_8County_PlusChampaign_2019_B19` (QL1), 4 × 1 km tiles.
- **CRS:** NAD83(2011) / Conus Albers metres (EPSG:6350); vertical NAVD88 Geoid12B (EPSG:5703).
- **Extent (WGS84):** `-88.2402753, 40.0990944, -88.2147506, 40.1183436` (centre ≈ UIUC Main Quad).
- The point cloud is **not** in this repo (336 MB); the notebook fetches it from I-GUIDE.
- **OSM 2019 (VGI layer):** Illinois statewide shapefile extracts, WGS84 —
  1,197,659 building polygons and 765,328 road lines with `osm_id`, `lastchange`,
  `fclass`, and county-join attributes. Too large for git (zips 138/108 MB), so they live
  as assets on [release `osm-il-2019`](https://github.com/rayford295/uiuc-campus-lidar/releases/tag/osm-il-2019).
  Campus-bbox subsets (`osm_buildings_2019.geojson`, `osm_roads_2019.geojson`) are
  committed at the repo root and feed `src/vgi_comparison.py` directly.

## Run it

```bash
pip install -r requirements.txt
jupyter lab UIUC_campus_LiDAR_pipeline.ipynb   # run all cells
```

Or the scripts directly (expect the `.laz` at the repo root):

```bash
python src/classical_detection.py   # ground/DTM, buildings, trees  -> outputs/detection/
python src/pointnet_semseg.py       # PointNet semantic seg (baseline) -> outputs/segmentation/
python src/dgcnn_semseg.py          # DGCNN semantic seg + full-tile map -> outputs/segmentation/
```

The segmentation scripts reuse `outputs/detection/dtm.tif` (height-above-ground) and a
cached feature set, so run `classical_detection.py` first on a fresh checkout.

Device is auto-selected **CUDA → Apple MPS → CPU**, so the notebook runs unchanged on a
GPU server or a laptop. Full run ≈ 10–15 min on CPU/MPS.

## Repository layout

```
UIUC_campus_LiDAR_pipeline.ipynb   reproducible end-to-end notebook (download → outputs)
src/
  classical_detection.py           ground/DTM + building & tree instances
  pointnet_semseg.py               PointNet semantic segmentation (baseline)
  dgcnn_semseg.py                  DGCNN (EdgeConv) semantic segmentation + full-tile map
  vgi_comparison.py                LiDAR-vs-reference building IoU matching + completeness map
  pixel_comparison.py              per-pixel building agreement (0.1/0.2 m) + disagreement map
  naip_segmentation.py             NAIP land cover + building (LiDAR-fused) + paved/road
metadata/
  georeference.txt  encoding.txt  footprint.geojson    CRS / bbox / ASPRS codes (GIS · OSM)
outputs/
  README.md                        methods & metrics write-up
  detection/                       classical-detection results
    buildings.geojson  trees.geojson                   (WGS84, for QGIS / JOSM)
    coverage_dsm.png  ground_dtm.png  *_detected.png    figures
    detection_summary.json
  segmentation/                    deep-learning segmentation results
    dl_*  (PointNet)   seg_*  (DGCNN)                   figures · metrics · weights
  comparison/                      VGI-vs-LiDAR building comparison (+ README caveats)
  naip/                            NAIP land-cover / building / paved segmentation
osm_buildings_2019.geojson         OSM 2019 buildings, campus bbox (VGI layer under evaluation)
osm_roads_2019.geojson             OSM 2019 roads, campus bbox
PROJECT_DESCRIPTION.md             research framing: evaluating & calibrating VGI with RS
requirements.txt   LICENSE
```
Large / regenerable artifacts (`*.laz`, `outputs/**/*.tif`, feature cache) are gitignored —
the notebook downloads the cloud and the scripts recreate the rest.

## Method (brief)

1. **Merge** — four adjacent tiles concatenated into one gap-free cloud (they share edges exactly).
2. **Rasterize** (0.5 m, single pass) — surface (max-Z, noise removed), terrain (min-Z of ground),
   class counts. `CHM = DSM − DTM` gives height-above-ground.
3. **Buildings** — class-6 raster → morphology → connected components; height from CHM.
4. **Trees** — smoothed-CHM local maxima (≥3 m, 3 m spacing) + watershed crowns.
5. **Semantic segmentation** — block-wise PointNet trained on the ASPRS labels with a
   **spatial** west/east train–val split (no leakage); height-above-ground is the key feature.

## License

Code: [MIT](LICENSE). Input LiDAR data: USGS 3DEP, U.S. public domain.

## Citation

Point cloud derived from *USGS 3DEP Lidar Point Cloud, IL_8County_PlusChampaign_2019_B19*,
U.S. Geological Survey (public domain).

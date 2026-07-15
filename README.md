# Detecting and Correcting Spatial Bias in VGI Using Remote Sensing

Volunteered Geographic Information (VGI) such as OpenStreetMap is accurate in well-mapped
urban areas but incomplete in data-sparse regions. This project evaluates and calibrates
VGI using multimodal remote sensing (LiDAR + aerial imagery), starting from the UIUC
campus and scaling toward an urban→rural gradient.

An [I-GUIDE Summer School 2026 project](https://i-guide.io/summer-school/summer-school-2026/summer-school-2026-projects/).
Research questions and full framing: [docs/PROJECT_DESCRIPTION.md](docs/PROJECT_DESCRIPTION.md) ·
comparison design: [docs/METHODOLOGY.md](docs/METHODOLOGY.md) · metrics: [docs/METRICS.md](docs/METRICS.md).

## Results

| | |
|---|---|
| ![buildings](results/detection/buildings_detected.png) | ![trees](results/detection/trees_detected.png) |
| **1,312 building instances** from LiDAR (footprint + height) | **11,777 individual trees** (height + crown) |
| ![fulltile](results/segmentation/seg_fulltile.png) | ![naip](results/naip/naip_segmentation.png) |
| **DGCNN** semantic segmentation, OA 0.930 · mIoU 0.768 | **NAIP** land cover, corroborating RS reference |

The remote-sensing reference comes from a reproducible pipeline over a merged 2 × 2 km
USGS 3DEP QL1 point cloud (80.8 M points): classical detection (buildings, trees, DTM)
plus DGCNN semantic segmentation of the ASPRS classes (PointNet baseline OA 0.913 /
mIoU 0.707 → DGCNN 0.930 / 0.768, spatial train/val split).

## VGI comparison — first result

LiDAR footprints (corroborated by NAIP) are the ground truth; OSM 2019 `building=*`
(temporally matched to the LiDAR) is evaluated against them — IoU matching →
completeness → gridded bias map:

```bash
python src/vgi_comparison.py data/osm_buildings_2019.geojson
```

![OSM vs LiDAR comparison](results/comparison/comparison_map.png)

| completeness (count) | completeness (area) | OSM commission | pixel IoU (0.2 m) | Cohen κ |
|---|---|---|---|---|
| **58.3%** | **79.4%** | 29.5% | 0.698 | 0.774 |

OSM captures the large institutional buildings (hence 79% by area) but misses 547 small
structures, and completeness collapses to **< 0.3 on the eastern residential strip** —
a sharp spatial-bias gradient inside a single 2 × 2 km tile.

**Temporal validation:** against current OSM (2026), completeness rises to 81.9% / 91.8%
— **64% of the 2019 gaps have since been filled by the community**, confirming they were
genuine omissions (the buildings were in the 2019 LiDAR all along), not yet-unbuilt
structures.

**Roads** (vs the NAIP paved layer — LiDAR has no road class): 91% of OSM 2019 way
length has pavement evidence, so roads were already well-mapped where buildings were not;
2026 adds micro-mapping detail (+60% segments, +8% length), and **80% of the added
length was already paved in the 2019 imagery** — filled gaps, mirroring the buildings
story. Unexplained pavement is mostly parking, and canopy-shaded streets are a known
optical false alarm. Details and caveats:
[results/comparison/](results/comparison/README.md).
Full Illinois statewide OSM (1.20 M buildings, 765 K roads) for scaling the gradient is
on the [`osm-il-2019` release](https://github.com/rayford295/vgi-spatial-bias/releases/tag/osm-il-2019).

## Quick start

```bash
pip install -r requirements.txt
jupyter lab UIUC_campus_LiDAR_pipeline.ipynb        # self-contained: downloads the cloud, runs all
```

Or run the scripts directly (in order — later ones reuse the DTM and feature cache):

```bash
python src/classical_detection.py    # ground/DTM, buildings, trees   -> results/detection/
python src/dgcnn_semseg.py           # semantic segmentation          -> results/segmentation/
python src/vgi_comparison.py data/osm_buildings_2019.geojson   # bias map -> results/comparison/
```

Device auto-selects CUDA → Apple MPS → CPU; a full run takes ≈ 10–15 min.

## Repository layout

```
UIUC_campus_LiDAR_pipeline.ipynb   end-to-end reproducible notebook
src/                               pipeline scripts (detection, segmentation, comparison, NAIP)
data/                              OSM 2019 campus subsets + CRS/bbox metadata
docs/                              PROJECT_DESCRIPTION · METHODOLOGY · METRICS
results/                           detection/ · segmentation/ · comparison/ · naip/  (+ write-up)
```

Large / regenerable artifacts (`.laz`, `.tif`, caches) are gitignored — the notebook
downloads the point cloud from I-GUIDE storage and the scripts recreate the rest.

## Data sources

- **LiDAR:** USGS 3DEP `IL_8County_PlusChampaign_2019_B19` (QL1), EPSG:6350 / NAVD88.
- **OSM 2019:** Illinois statewide shapefiles (WGS84) — see the release above.
- **NAIP:** 4-band aerial imagery, ~0.7 m, fused with LiDAR for the impervious/paved layers.

## License

MIT for code (see [LICENSE](LICENSE)); USGS 3DEP data is public domain.

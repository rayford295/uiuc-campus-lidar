# Outputs — methods & metrics

Input: `UIUC_campus_LiDAR_merged_2x2km.laz` (80.8 M pts, EPSG:6350, NAVD88 heights).
Reproduce with the scripts in [`../src/`](../src) (run `classical_detection.py` first —
the segmentation scripts reuse its `detection/dtm.tif`).

## detection/ — classical instance detection (`src/classical_detection.py`, CPU, ~2 min)
One 0.5 m rasterization pass feeds three detectors.

| Target | Method | Result | Files |
|--------|--------|--------|-------|
| **Ground / DTM** | min-Z of class 2 + nearest-fill | bare-earth terrain | `dtm.tif`, `chm.tif`, `ground_dtm.png` |
| **Buildings** | class-6 raster → morphology → connected components; height from CHM | **1,312 buildings**, 0.94 km² footprint, median 8.3 m | `buildings.geojson`, `buildings_detected.png` |
| **Trees** | CHM local-maxima (≥3 m, 3 m spacing) + watershed crowns | **11,777 trees**, median 12 m, 21.5 m² crown | `trees.geojson`, `trees_detected.png` |

`coverage_dsm.png` is the 1 m DSM hillshade used to confirm gap-free coverage.
GeoJSON is WGS84 lon/lat (EPSG:4326) — drop straight into QGIS / JOSM / iD.
Attributes: buildings → `area_m2`, `height_m`; trees → `height_m`, `crown_m2`.

## segmentation/ — deep-learning semantic segmentation (PyTorch, CUDA→MPS→CPU)
Predict the 5 ASPRS classes; ground truth = the LAZ classification. **Spatial split**
(west train / east val) — no leakage. Key feature: **height-above-ground** (z − DTM),
which removes the 14 m campus elevation gradient (ground≈0, buildings=plateau, trees=tall).

| Model | script | OA | mIoU | Ground | Low Veg | Med Veg | High Veg | Building |
|-------|--------|----|------|--------|---------|---------|----------|----------|
| PointNet (baseline) | `pointnet_semseg.py` | 0.913 | 0.707 | 0.96 | 0.39 | 0.57 | 0.85 | 0.78 |
| **DGCNN (EdgeConv)** | `dgcnn_semseg.py` | **0.930** | **0.768** | 0.95 | **0.49** | **0.70** | **0.88** | **0.82** |

- PointNet files: `dl_prediction.png`, `dl_confusion.png`, `dl_metrics.json`, `dl_pointnet.pt`.
- DGCNN files: `seg_fulltile.png` (wall-to-wall map), `seg_confusion.png`, `seg_metrics.json`,
  `seg_dgcnn.pt`, `seg_labels.tif` (0.5 m label raster, gitignored).

**Why DGCNN wins:** its dynamic k-NN graph gives each point local geometric context, so
building edges sharpen and every vegetation class improves. Remaining errors are
"adjacent-class" only (Low Veg↔Ground, Med↔High Veg); Building↔Ground ≈ 0.01. The
full-tile map shows no discontinuity at the train|val boundary.

### Next steps
- Per-point labeled LAZ (propagate predictions to all 80 M points).
- DL *instance* segmentation (offset head + clustering) for individual buildings/trees.
- Fold the ~30 % "Unclassified" points into a 6-class problem to label the whole cloud.

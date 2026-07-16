"""Proposal confidence: rule tiers (approach A) vs learned scorer (approach C).

Both answer the same question — "how much should a mapper trust this proposed
building?" — and both are evaluated against the same objective label: did the
OSM community actually map this gap by 2026?

  A (rules)   tier 0-3 = one point each for area >= 80 m2, height >= 4 m,
              NAIP impervious fraction >= 0.5
  C (learned) gradient-boosted classifier on 6 features:
              area, height, impervious fraction, distance to nearest 2019 road,
              100 m neighborhood OSM completeness, shape fill ratio

Spatial split at the tile midline (west = train, east = eval), matching the
DGCNN protocol, so the learned scorer is never evaluated where it trained.

Outputs (results/correction/):
  scores.csv            per gap: features, tier score, GBM probability, label
  scorer_summary.json   AUC / precision@50 for both approaches + importances

Usage:  python src/acceptance_scorer.py
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from propose_geometry import best_match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGION = sys.argv[1] if len(sys.argv) > 1 else "uiuc_campus"
DATA = os.path.join(ROOT, "data", REGION)
RES = os.path.join(ROOT, "results", REGION)
OUT = os.path.join(RES, "correction")
# fixed west/east split line per region (tile midline, same as the DGCNN split)
SPLIT_X_BY_REGION = {"uiuc_campus": 656000.0, "colorado_springs": -752834.5}
CRS, NEIGH_M, TOP_K = 6350, 100.0, 50


def raster_fraction(src, band, geoms):
    """Mean of a binary raster under each geometry (windowed reads)."""
    out = []
    for geom in geoms:
        b = geom.bounds
        try:
            win = from_bounds(*b, src.transform).round_offsets().round_lengths()
        except Exception:
            out.append(np.nan)
            continue
        r0, c0 = max(0, win.row_off), max(0, win.col_off)
        r1 = min(src.height, win.row_off + win.height)
        c1 = min(src.width, win.col_off + win.width)
        if r1 <= r0 or c1 <= c0:
            out.append(np.nan)
            continue
        sub = band[r0:r1, c0:c1]
        m = ~geometry_mask([geom], out_shape=sub.shape,
                           transform=rasterio.windows.transform(
                               rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0),
                               src.transform))
        out.append(float(sub[m].mean()) if m.any() else np.nan)
    return np.array(out)


def build_features(gaps):
    # NAIP impervious = building OR paved (both binary rasters from stage 3)
    with rasterio.open(os.path.join(RES, "naip", "naip_building.tif")) as b, \
         rasterio.open(os.path.join(RES, "naip", "naip_paved.tif")) as p:
        imperv = ((b.read(1) > 0) | (p.read(1) > 0)).astype("uint8")
        gaps["imperv_frac"] = raster_fraction(b, imperv, gaps.geometry)

    roads = gpd.read_file(os.path.join(DATA, "osm_roads_2019.geojson")).to_crs(CRS)
    try:
        _, dist = roads.sindex.nearest(gaps.geometry, return_distance=True,
                                       return_all=False)
        gaps["dist_road"] = dist
    except TypeError:  # old geopandas: brute force against the merged network
        net = roads.geometry.unary_union
        gaps["dist_road"] = gaps.geometry.apply(lambda g: g.distance(net))

    # neighborhood completeness: share of LiDAR buildings within 100 m that
    # OSM had mapped in 2019 (contributor attention around the gap)
    lid = gpd.read_file(os.path.join(RES, "detection", "buildings.geojson")).to_crs(CRS)
    lid["geometry"] = lid.geometry.buffer(0)
    o19 = gpd.read_file(os.path.join(DATA, "osm_buildings_2019.geojson")).to_crs(CRS)
    o19["geometry"] = o19.geometry.buffer(0)
    o19 = o19[o19.area >= 5.0].reset_index(drop=True)
    iou, _ = best_match(lid, o19)
    lid["in19"] = iou >= 0.3

    buf = gpd.GeoDataFrame(geometry=gaps.geometry.centroid.buffer(NEIGH_M), crs=CRS)
    joined = gpd.sjoin(buf, lid[["in19", "geometry"]],
                       predicate="intersects", how="left")
    comp = joined.groupby(level=0)["in19"].mean()
    gaps["neigh_completeness"] = comp.reindex(range(len(gaps))).fillna(0)

    mrr_area = gaps.geometry.apply(
        lambda g: g.minimum_rotated_rectangle.area if not g.is_empty else np.nan)
    gaps["fill_ratio"] = (gaps.geometry.area / mrr_area).clip(0, 1)
    return gaps


FEATURES = ["area_m2", "height_m", "imperv_frac", "dist_road",
            "neigh_completeness", "fill_ratio"]


def precision_at_k(labels, scores, k=TOP_K):
    order = np.argsort(-np.asarray(scores))
    return float(np.asarray(labels)[order[:k]].mean())


def main():
    gaps = gpd.read_file(os.path.join(OUT, "gaps_labeled.geojson")).to_crs(CRS)
    gaps = build_features(gaps)
    gaps["tier"] = ((gaps.area_m2 >= 80).astype(int) +
                    (gaps.height_m >= 4).astype(int) +
                    (gaps.imperv_frac.fillna(0) >= 0.5).astype(int))

    X = gaps[FEATURES].fillna(gaps[FEATURES].median())
    y = gaps["filled"].astype(int)
    train, test = ~gaps["east"], gaps["east"]

    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=42)
    gbm.fit(X[train], y[train])
    gaps["gbm_prob"] = gbm.predict_proba(X)[:, 1]

    res = dict(
        train_gaps=int(train.sum()), train_filled=int(y[train].sum()),
        eval_gaps=int(test.sum()), eval_filled=int(y[test].sum()),
        auc_rule_tiers=round(float(roc_auc_score(y[test], gaps.tier[test])), 3),
        auc_gbm=round(float(roc_auc_score(y[test], gaps.gbm_prob[test])), 3),
        precision_at_50_rule=round(precision_at_k(
            y[test].values, (gaps.tier[test] * 1e6 + gaps.area_m2[test]).values), 3),
        precision_at_50_gbm=round(precision_at_k(
            y[test].values, gaps.gbm_prob[test].values), 3),
        base_rate_eval=round(float(y[test].mean()), 3),
        gbm_feature_importance={f: round(float(v), 3) for f, v in
                                zip(FEATURES, gbm.feature_importances_)})

    cols = ["gap_id", "filled", "east", "tier", "gbm_prob"] + FEATURES
    gaps[cols].round(4).to_csv(os.path.join(OUT, "scores.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "scorer_summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

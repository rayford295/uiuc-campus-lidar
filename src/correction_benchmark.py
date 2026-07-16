"""Head-to-head benchmark of the three correction approaches.

Everything is evaluated on the held-out east half against what the OSM
community actually drew by 2026:

  geometry (filled gaps only)  IoU / centroid offset / area ratio vs the
                               community polygon — rule (A=C) vs learned (B),
                               plus the raw LiDAR footprint as the no-
                               regularization baseline
  confidence (all east gaps)   does the score predict community acceptance?
                               AUC + precision@50 — rule tiers (A) vs GBM (C)
                               vs U-Net mask probability (B)

Figures: proposal gallery (truth vs A vs B), deployment map of the 195
still-unmapped gaps ranked by the best scorer, and the summary panel.

Usage:  python src/correction_benchmark.py
Writes results/correction/correction_benchmark.json + 3 figures.
"""
import json
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "correction")
CRS = 6350
C_RULE, C_LEARN, C_TRUTH, C_RAW = "#2a78d6", "#eb6834", "#008300", "#b0aea5"


def iou(a, b):
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter) if inter else 0.0


def geometry_metrics(props, truth):
    rows = []
    for _, p in props.iterrows():
        t = truth.get(p.gap_id)
        if t is None or p.geometry.is_empty:
            continue
        rows.append(dict(gap_id=p.gap_id, iou=iou(p.geometry, t),
                         centroid_off=p.geometry.centroid.distance(t.centroid),
                         area_ratio=p.geometry.area / t.area))
    df = pd.DataFrame(rows)
    return df, dict(n=int(len(df)), median_iou=round(float(df.iou.median()), 3),
                    mean_iou=round(float(df.iou.mean()), 3),
                    median_centroid_off_m=round(float(df.centroid_off.median()), 2),
                    median_area_ratio=round(float(df.area_ratio.median()), 2))


def precision_at_k(labels, scores, k=50):
    order = np.argsort(-np.asarray(scores, dtype=float))
    return round(float(np.asarray(labels)[order[:k]].mean()), 3)


def main():
    rule = gpd.read_file(os.path.join(OUT, "proposals_rule.geojson")).to_crs(CRS)
    learned = gpd.read_file(os.path.join(OUT, "proposals_learned.geojson")).to_crs(CRS)
    gaps = gpd.read_file(os.path.join(OUT, "gaps_labeled.geojson")).to_crs(CRS)
    truth_gdf = gpd.read_file(os.path.join(OUT, "community_truth.geojson")).to_crs(CRS)
    truth = {r.gap_id: r.geometry.buffer(0) for _, r in truth_gdf.iterrows()}
    scores = pd.read_csv(os.path.join(OUT, "scores.csv"))

    east_filled = gaps[gaps.east & gaps.filled].gap_id.values
    sel = lambda g: g[g.gap_id.isin(east_filled)]

    geo = {}
    geo_df = {}
    geo_df["rule (A/C)"], geo["rule (A/C)"] = geometry_metrics(sel(rule), truth)
    geo_df["learned (B)"], geo["learned (B)"] = geometry_metrics(sel(learned), truth)
    geo_df["raw LiDAR"], geo["raw LiDAR"] = geometry_metrics(sel(gaps.assign(
        gap_id=gaps.gap_id)), truth)

    # confidence: east gaps, label = filled
    se = scores[scores.east].merge(
        learned[["gap_id", "confidence"]], on="gap_id", how="left")
    y = se.filled.astype(int).values
    conf = dict(
        auc=dict(rule_tiers=round(float(roc_auc_score(y, se.tier)), 3),
                 gbm=round(float(roc_auc_score(y, se.gbm_prob)), 3),
                 unet_mask=round(float(roc_auc_score(y, se.confidence.fillna(0))), 3)),
        precision_at_50=dict(
            rule_tiers=precision_at_k(y, se.tier * 1e6 + se.area_m2),
            gbm=precision_at_k(y, se.gbm_prob),
            unet_mask=precision_at_k(y, se.confidence.fillna(0))),
        base_rate=round(float(y.mean()), 3), eval_gaps=int(len(se)))

    res = dict(geometry_east_filled=geo, confidence_east=conf)
    json.dump(res, open(os.path.join(OUT, "correction_benchmark.json"), "w"),
              indent=2)
    print(json.dumps(res, indent=2))

    # ---- figure 1: proposal gallery --------------------------------------
    ex = sel(rule).merge(sel(learned)[["gap_id"]], on="gap_id")
    ex_ids = (gaps[gaps.gap_id.isin(ex.gap_id)]
              .sort_values("area_m2").gap_id.values)
    picks = ex_ids[np.linspace(2, len(ex_ids) - 3, 8).astype(int)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, gid in zip(axes.ravel(), picks):
        g = gaps[gaps.gap_id == gid].iloc[0]
        t = truth[gid]
        gpd.GeoSeries([g.geometry]).plot(ax=ax, color=C_RAW, alpha=0.4)
        for gdf, c, lw in ((truth_gdf[truth_gdf.gap_id == gid], C_TRUTH, 2.2),
                           (rule[rule.gap_id == gid], C_RULE, 1.8),
                           (learned[learned.gap_id == gid], C_LEARN, 1.8)):
            gdf.boundary.plot(ax=ax, color=c, linewidth=lw)
        b = t.buffer(12).bounds
        ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
        ax.set_title(f"gap {gid} · {g.area_m2:.0f} m²", fontsize=9)
        ax.set_axis_off()
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=2, label=l) for c, l in
                        ((C_TRUTH, "community 2026 (truth)"),
                         (C_RULE, "rule proposal (A/C)"),
                         (C_LEARN, "learned proposal (B)"),
                         (C_RAW, "raw LiDAR footprint"))],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Proposed corrections vs what the community drew — held-out east half")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "proposal_gallery.png"), dpi=130,
                facecolor="white")
    plt.close(fig)

    # ---- figure 2: deployment map (still-unmapped, ranked) ---------------
    todo = gaps[~gaps.filled].merge(scores[["gap_id", "gbm_prob"]], on="gap_id")
    o19 = gpd.read_file(os.path.join(ROOT, "data",
                                     "osm_buildings_2019.geojson")).to_crs(CRS)
    fig, ax = plt.subplots(figsize=(10, 10))
    o19.plot(ax=ax, color="#e8e6dc")
    todo.plot(ax=ax, column="gbm_prob", cmap="Blues", legend=True,
              legend_kwds={"shrink": 0.6, "label": "acceptance score (GBM)"})
    ax.set_title(f"Deployment: {len(todo)} still-unmapped buildings, "
                 "ranked by learned acceptance score")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "deployment_map.png"), dpi=130,
                facecolor="white")
    plt.close(fig)

    # ---- figure 3: summary panel ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(geo_df)
    axes[0].boxplot([geo_df[n].iou for n in names], showfliers=False)
    axes[0].set_xticks(range(1, len(names) + 1))
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("IoU vs community polygon")
    axes[0].set_title("Geometry quality (east filled gaps)")
    labels = ["rule tiers (A)", "GBM (C)", "U-Net (B)"]
    keys = ["rule_tiers", "gbm", "unet_mask"]
    x = np.arange(3)
    axes[1].bar(x - 0.18, [conf["auc"][k] for k in keys], 0.36,
                color=C_RULE, label="AUC")
    axes[1].bar(x + 0.18, [conf["precision_at_50"][k] for k in keys], 0.36,
                color=C_LEARN, label="precision@50")
    axes[1].axhline(conf["base_rate"], color="0.4", ls="--", lw=1,
                    label=f"base rate {conf['base_rate']}")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Confidence quality (predicting community acceptance)")
    axes[1].legend(frameon=False, fontsize=9)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "benchmark_summary.png"), dpi=130,
                facecolor="white")
    plt.close(fig)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()

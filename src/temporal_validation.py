"""
Temporal validation: did the community fill the 2019 OSM gaps by 2026?

Classifies every LiDAR-detected building (2019 ground truth) as:
  - mapped in OSM 2019            (green)
  - gap filled by OSM 2026        (blue)   <- validates the gap was a real omission
  - still unmapped in OSM 2026    (red)    <- persistent omission
A 2019 gap that today's OSM has filled cannot be explained by "the building didn't
exist yet" (the LiDAR flew in 2019) — it was community under-mapping, which is exactly
the spatial bias this project measures.

Usage:  python src/temporal_validation.py
Reads data/osm_buildings_2019.geojson + data/osm_buildings_2026.geojson.
Writes results/comparison/temporal/ (summary JSON + evolution map).
"""
import os, json, numpy as np, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# optional region args: <lidar.geojson> <osm_old.geojson> <osm_new.geojson> <out_dir>
_A = sys.argv[1:]
OUT = _A[3] if len(_A) > 3 else os.path.join(ROOT, "results", "comparison", "temporal")
os.makedirs(OUT, exist_ok=True)
CRS, IOU_THR, MIN_AREA, GRID = 6350, 0.3, 5.0, 250.0

def load(path):
    g = gpd.read_file(path).to_crs(CRS)
    g["geometry"] = g.geometry.buffer(0)
    return g[g.area >= MIN_AREA].reset_index(drop=True)

def best_iou(lid, ref):
    pairs = gpd.sjoin(lid[["geometry"]], ref[["geometry"]], predicate="intersects", how="inner")
    if pairs.empty:
        return np.zeros(len(lid))
    inter = pairs.geometry.values.intersection(ref.geometry.values[pairs.index_right.values]).area
    iou = inter / (lid.area.values[pairs.index.values] + ref.area.values[pairs.index_right.values] - inter)
    return pairs.assign(iou=iou).groupby(level=0)["iou"].max().reindex(range(len(lid))).fillna(0).values

lid = load(_A[0] if _A else os.path.join(ROOT, "results", "detection", "buildings.geojson"))
o19 = load(_A[1] if len(_A) > 1 else os.path.join(ROOT, "data", "osm_buildings_2019.geojson"))
o26 = load(_A[2] if len(_A) > 2 else os.path.join(ROOT, "data", "osm_buildings_2026.geojson"))

lid["in19"] = best_iou(lid, o19) >= IOU_THR
lid["in26"] = best_iou(lid, o26) >= IOU_THR
lid["status"] = np.select(
    [lid.in19, ~lid.in19 & lid.in26], ["mapped_2019", "filled_by_2026"], "still_unmapped")

n = len(lid); a = lid.area
summary = dict(
    lidar_buildings=int(n),
    osm2019_buildings=int(len(o19)), osm2026_buildings=int(len(o26)),
    mapped_2019=int(lid.in19.sum()),
    gaps_2019=int((~lid.in19).sum()),
    filled_by_2026=int(((~lid.in19) & lid.in26).sum()),
    still_unmapped=int((lid.status == "still_unmapped").sum()),
    gap_fill_rate=round(float(((~lid.in19) & lid.in26).sum() / (~lid.in19).sum()), 4),
    completeness_2019_count=round(float(lid.in19.mean()), 4),
    completeness_2026_count=round(float(lid.in26.mean()), 4),
    completeness_2019_area=round(float(a[lid.in19].sum() / a.sum()), 4),
    completeness_2026_area=round(float(a[lid.in26].sum() / a.sum()), 4))
json.dump(summary, open(os.path.join(OUT, "temporal_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))

COLORS = {"mapped_2019": "#4c9f70", "filled_by_2026": "#3b6fb0", "still_unmapped": "#d1483f"}
LABELS = {"mapped_2019": f"mapped in OSM 2019 ({summary['mapped_2019']})",
          "filled_by_2026": f"gap filled by 2026 ({summary['filled_by_2026']})",
          "still_unmapped": f"still unmapped ({summary['still_unmapped']})"}
fig, ax = plt.subplots(figsize=(11, 10))
for s, c in COLORS.items():
    sel = lid[lid.status == s]
    if len(sel): sel.plot(ax=ax, color=c, ec="none")
ax.legend(handles=[Patch(color=c, label=LABELS[s]) for s, c in COLORS.items()],
          loc="lower center", ncol=3, fontsize=9)
ax.set_title("OSM completeness evolution 2019 → 2026 vs 2019 LiDAR\n"
             f"count completeness {summary['completeness_2019_count']:.1%} → "
             f"{summary['completeness_2026_count']:.1%};  "
             f"{summary['gap_fill_rate']:.0%} of 2019 gaps since filled by the community")
ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "temporal_evolution.png"), dpi=130); plt.close(fig)
print("done ->", OUT)

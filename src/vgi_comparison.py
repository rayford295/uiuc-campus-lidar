"""
VGI-vs-LiDAR building comparison — the core "detect spatial bias in OSM" step.

Matches LiDAR-detected building footprints (our detection, the objective ground truth)
against a REFERENCE building layer (ultimately OSM building=*; here a reference shapefile
passed as argv[1]) via IoU, then reports:
  - completeness (recall)  = reference captured / LiDAR reality      -> OSM omissions
  - commission             = reference-only / reference total        -> OSM extras / stale
  - a gridded completeness surface                                   -> the spatial-bias map
  - omissions.geojson      = LiDAR buildings absent from reference    -> the "correction" layer

Usage:  python src/vgi_comparison.py <reference.shp|.geojson> [output_subdir]
Both layers are reprojected to EPSG:6350 (equal-area metres) before matching.
With output_subdir, results go to results/comparison/<output_subdir>/ so runs against
different references (e.g. OSM 2019 vs current OSM) can coexist.

NOTE: with a LiDAR-derived reference this is a cross-method check / pipeline test,
NOT the OSM-vs-LiDAR result. Swap argv[1] for real OSM-2019 buildings for the study.
"""
import os, sys, json, numpy as np, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "comparison", *(sys.argv[2:3])); os.makedirs(OUT, exist_ok=True)
LIDAR = os.path.join(ROOT, "results", "detection", "buildings.geojson")
REF = sys.argv[1] if len(sys.argv) > 1 else None
if not REF or not os.path.exists(REF):
    sys.exit("usage: python src/vgi_comparison.py <reference_buildings.shp|.geojson>\n"
             "  reference = OSM building=* polygons (study) or any building layer to compare.")
CRS = 6350            # equal-area metres
IOU_THR = 0.3         # min IoU to call a LiDAR building "mapped" in the reference
GRID = 250.0          # m, completeness-surface cell size
MIN_AREA = 5.0        # m^2, drop slivers

def load(path):
    g = gpd.read_file(path).to_crs(CRS)
    g["geometry"] = g.geometry.buffer(0)                 # fix invalid/zero-area geoms
    g = g[g.area >= MIN_AREA].reset_index(drop=True)
    return g

lid = load(LIDAR); ref = load(REF)
lid["a"] = lid.area; ref["a"] = ref.area
print(f"LiDAR buildings: {len(lid)}   reference: {len(ref)}")

# ---- IoU matching via spatial join on intersection ----
pairs = gpd.sjoin(lid[["geometry"]], ref[["geometry"]],
                  predicate="intersects", how="inner")
gref = ref.geometry.values
inter = pairs.geometry.values.intersection(gref[pairs.index_right.values]).area
a_lid = lid["a"].values[pairs.index.values]
a_ref = ref["a"].values[pairs.index_right.values]
iou = inter / (a_lid + a_ref - inter)
pairs = pairs.assign(iou=iou)

best_lid = pairs.groupby(level=0)["iou"].max()            # best IoU per LiDAR building
best_ref = pairs.groupby("index_right")["iou"].max()      # best IoU per reference building
lid["iou"] = best_lid.reindex(range(len(lid))).fillna(0).values
ref["iou"] = best_ref.reindex(range(len(ref))).fillna(0).values
lid["matched"] = lid["iou"] >= IOU_THR
ref["matched"] = ref["iou"] >= IOU_THR

# ---- summary metrics ----
completeness = lid["matched"].mean()
area_completeness = lid.loc[lid.matched,"a"].sum() / lid["a"].sum()
commission = 1 - ref["matched"].mean()
summary = dict(
    reference=os.path.basename(REF), iou_threshold=IOU_THR,
    lidar_buildings=int(len(lid)), reference_buildings=int(len(ref)),
    matched=int(lid["matched"].sum()),
    completeness_count=round(float(completeness),4),          # recall of reference vs LiDAR
    completeness_area=round(float(area_completeness),4),
    reference_commission=round(float(commission),4),          # reference features w/o LiDAR support
    lidar_only_omissions=int((~lid["matched"]).sum()),
    reference_only=int((~ref["matched"]).sum()),
    median_matched_iou=round(float(lid.loc[lid.matched,"iou"].median()),3))
json.dump(summary, open(os.path.join(OUT,"comparison_summary.json"),"w"), indent=2)
print(json.dumps(summary, indent=2))

# ---- omissions layer (LiDAR buildings missing from reference) -> WGS84 ----
lid.loc[~lid.matched].to_crs(4326).to_file(os.path.join(OUT,"omissions.geojson"), driver="GeoJSON")

# ---- gridded completeness surface (spatial-bias map) ----
xmin, ymin, xmax, ymax = lid.total_bounds
nx = int(np.ceil((xmax-xmin)/GRID)); ny = int(np.ceil((ymax-ymin)/GRID))
cent = lid.geometry.centroid
cx = ((cent.x - xmin)/GRID).astype(int).clip(0, nx-1)
cy = ((cent.y - ymin)/GRID).astype(int).clip(0, ny-1)
tot = np.zeros((ny, nx)); mat = np.zeros((ny, nx))
for cyi, cxi, ar, m in zip(cy, cx, lid["a"], lid["matched"]):
    tot[cyi, cxi] += ar
    if m: mat[cyi, cxi] += ar
comp = np.where(tot > 0, mat/tot, np.nan)

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(19, 9))
lid.loc[lid.matched].plot(ax=ax[0], color="#4c9f70", ec="none")
lid.loc[~lid.matched].plot(ax=ax[0], color="#d1483f", ec="none")     # OSM omissions
ref.loc[~ref.matched].boundary.plot(ax=ax[0], color="#3b6fb0", lw=0.6)  # reference-only
ax[0].set_title(f"Building match — completeness {completeness:.1%} (count), {area_completeness:.1%} (area)")
ax[0].legend(handles=[Patch(color="#4c9f70",label="matched"),
                      Patch(color="#d1483f",label="LiDAR-only (reference omission)"),
                      Patch(ec="#3b6fb0",fc="none",label="reference-only")], loc="lower center", ncol=3)
ax[0].set_aspect("equal"); ax[0].ticklabel_format(style="plain")
im = ax[1].imshow(comp, origin="lower", extent=[xmin,xmax,ymin,ymax], cmap="RdYlGn",
                  vmin=0, vmax=1, aspect="equal")
ax[1].set_title(f"Reference completeness surface ({GRID:.0f} m grid) — the spatial-bias map")
plt.colorbar(im, ax=ax[1], shrink=.7, label="fraction of LiDAR building area captured")
ax[1].ticklabel_format(style="plain")
fig.suptitle(f"VGI vs LiDAR building comparison  —  reference: {os.path.basename(REF)}", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"comparison_map.png"), dpi=130); plt.close(fig)
print("done ->", OUT)

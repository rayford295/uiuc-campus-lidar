"""
Road network evolution 2019 → 2026 — the roads analogue of temporal_validation.py.

Classifies every OSM 2026 way (clipped to the NAIP tile) as:
  - existed in OSM 2019                       (grey)
  - added by 2026, paved in 2019 NAIP         (blue)  <- gap filled: the way physically
                                                         existed in 2019 but was unmapped
  - added by 2026, no 2019 pavement evidence  (red)   <- new construction, unpaved way,
                                                         or canopy-occluded (optical limit)
Because the NAIP imagery is contemporaneous with the 2019 snapshot, "added + paved" is
direct evidence the 2019 network was incomplete, mirroring the building gap-fill logic.
Also reports 2019 ways with no 2026 counterpart (deleted / redrawn).

Usage:  python src/road_evolution.py
Writes results/comparison/temporal/ (roads_evolution.png + roads_temporal_summary.json).
"""
import os, json, numpy as np, geopandas as gpd, rasterio, rasterio.plot
from shapely.geometry import box as shapely_box
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "comparison", "temporal"); os.makedirs(OUT, exist_ok=True)
PAVED = os.path.join(ROOT, "results", "naip", "naip_paved.tif")
MATCH_TOL = 5.0        # m, buffer around the 2019 network for "existed in 2019"
MATCH_FRAC = 0.5       # min fraction of a way's length inside that buffer
SUPPORT_THR = 0.5      # min 2019-paved fraction to call an added way "paved in 2019"
WIDTH = {"motorway": 15, "trunk": 14, "primary": 12, "secondary": 10, "tertiary": 9,
         "residential": 8, "unclassified": 7, "service": 5, "pedestrian": 4,
         "track": 3, "cycleway": 3, "footway": 2.5, "path": 2, "steps": 2}
DEFAULT_W = 4.0

src = rasterio.open(PAVED)
paved = src.read(1)

def load(path):
    g = gpd.read_file(path).to_crs(src.crs)
    g = gpd.clip(g, shapely_box(*src.bounds)).explode(index_parts=False)
    return g[g.geometry.length > 1].reset_index(drop=True)

def paved_fraction(geom):
    b = geom.bounds
    win = from_bounds(b[0], b[1], b[2], b[3], src.transform).round_offsets().round_lengths()
    r0, c0 = max(0, win.row_off), max(0, win.col_off)
    r1 = min(paved.shape[0], win.row_off + win.height); c1 = min(paved.shape[1], win.col_off + win.width)
    if r1 <= r0 or c1 <= c0:
        return np.nan
    sub = paved[r0:r1, c0:c1]
    m = ~geometry_mask([geom], out_shape=sub.shape,
                       transform=rasterio.windows.transform(
                           rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), src.transform))
    return float(sub[m].mean()) if m.any() else np.nan

def union_all(gdf):
    """geopandas >= 1.0 renamed unary_union -> union_all; support both
    (the I-GUIDE CyberGISX kernel ships Python 3.8 / geopandas <= 0.13)."""
    return gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union

g19 = load(os.path.join(ROOT, "data", "osm_roads_2019.geojson"))
g26 = load(os.path.join(ROOT, "data", "osm_roads_2026.geojson"))
net19 = union_all(g19).buffer(MATCH_TOL)
net26 = union_all(g26).buffer(MATCH_TOL)

g26["frac_in19"] = g26.geometry.intersection(net19).length / g26.geometry.length
g26["existed"] = g26["frac_in19"] >= MATCH_FRAC
added = g26[~g26.existed].copy()
added["w"] = added["fclass"].map(lambda fc: WIDTH.get(str(fc).split("_")[0], DEFAULT_W))
added["support19"] = [paved_fraction(r.geometry.buffer(r.w / 2 + 1.5)) for r in added.itertuples()]
added["paved19"] = added["support19"] >= SUPPORT_THR
g26.loc[added.index, "paved19"] = added["paved19"]
g26["status"] = np.select(
    [g26.existed, ~g26.existed & g26.get("paved19", False).fillna(False)],
    ["existed_2019", "added_paved2019"], "added_no_evidence")

g19["frac_in26"] = g19.geometry.intersection(net26).length / g19.geometry.length
g19["gone"] = g19["frac_in26"] < MATCH_FRAC

km = lambda s: round(float(s.geometry.length.sum() / 1000), 2)
add_pav, add_no = g26[g26.status == "added_paved2019"], g26[g26.status == "added_no_evidence"]
by_class = {fc: dict(n=int(len(s)), km=km(s))
            for fc, s in added.groupby("fclass") if len(s) >= 10}
summary = dict(
    ways_2019=int(len(g19)), ways_2026=int(len(g26)),
    km_2019=km(g19), km_2026=km(g26),
    existed_2019=int(g26.existed.sum()), km_existed=km(g26[g26.existed]),
    added_by_2026=int((~g26.existed).sum()), km_added=km(added),
    added_paved_in_2019=int(len(add_pav)), km_added_paved=km(add_pav),
    added_no_2019_evidence=int(len(add_no)), km_added_no_evidence=km(add_no),
    gap_fill_share_of_added_km=round(km(add_pav) / max(km(added), 1e-9), 4),
    ways_2019_gone_by_2026=int(g19.gone.sum()), km_gone=km(g19[g19.gone]),
    added_by_class=by_class,
    match_tolerance_m=MATCH_TOL, support_threshold=SUPPORT_THR)
json.dump(summary, open(os.path.join(OUT, "roads_temporal_summary.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "added_by_class"}, indent=2))

ext = rasterio.plot.plotting_extent(src)
fig, ax = plt.subplots(figsize=(11, 10))
ax.imshow(np.where(paved == 1, 0.08, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1)
g26[g26.status == "existed_2019"].plot(ax=ax, color="0.55", lw=0.6)
add_pav.plot(ax=ax, color="#3b6fb0", lw=1.4)
add_no.plot(ax=ax, color="#d1483f", lw=1.4)
ax.legend(handles=[
    Patch(color="0.55", label=f"in OSM 2019 ({summary['km_existed']:.0f} km)"),
    Patch(color="#3b6fb0", label=f"added by 2026, paved in 2019 ({summary['km_added_paved']:.0f} km — filled gaps)"),
    Patch(color="#d1483f", label=f"added, no 2019 pavement evidence ({summary['km_added_no_evidence']:.0f} km)")],
    loc="lower center", ncol=2, fontsize=9)
ax.set_title("OSM road network evolution 2019 → 2026 (vs 2019 NAIP pavement)\n"
             f"{summary['km_added']:.0f} km added; {summary['gap_fill_share_of_added_km']:.0%} of added length "
             "already paved in 2019 — mapping caught up with reality")
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "roads_evolution.png"), dpi=130); plt.close(fig)
print("done ->", OUT)

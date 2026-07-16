"""
OSM roads vs NAIP paved surface — the roads half of the VGI comparison.

LiDAR has no road class, so the RS reference for roads is the NAIP paved layer
(impervious − buildings, from src/naip_segmentation.py). Two directions:

  forward  — per OSM segment: buffer the centreline by a class-dependent width and
             measure the fraction of the buffer that is NAIP-paved ("RS support").
             Low support = OSM road without pavement evidence (error, unpaved, or new).
  reverse  — fraction of paved pixels within any (buffered) OSM way = "explained".
             Unexplained paved = candidate unmapped ways — NOTE: paved is a superset of
             roads (parking lots, plazas), so unexplained ≠ automatically missing roads.

Runs on both the 2019 snapshot and the current (2026) snapshot for temporal validation.

Usage:  python src/road_comparison.py
Writes results/comparison/roads/ (summary JSON, per-year maps).
"""
import os, json, numpy as np, geopandas as gpd, rasterio
from shapely.geometry import box as shapely_box
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask, rasterize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "comparison", "roads"); os.makedirs(OUT, exist_ok=True)
PAVED = os.path.join(ROOT, "results", "naip", "naip_paved.tif")
SUPPORT_THR = 0.5          # min paved fraction in buffer to call a segment "supported"
TOL = 1.5                  # m, extra buffer for positional tolerance in the reverse test
WIDTH = {"motorway": 15, "trunk": 14, "primary": 12, "secondary": 10, "tertiary": 9,
         "residential": 8, "unclassified": 7, "service": 5, "pedestrian": 4,
         "track": 3, "cycleway": 3, "footway": 2.5, "path": 2, "steps": 2}
DEFAULT_W = 4.0

src = paved = paved_area_px = None


def init(paved_path=PAVED, out_dir=None):
    """Lazy raster setup so other regions can point at their own paved layer."""
    global src, paved, paved_area_px, OUT
    if out_dir:
        OUT = out_dir
    os.makedirs(OUT, exist_ok=True)
    src = rasterio.open(paved_path)
    paved = src.read(1)
    paved_area_px = float(abs(src.res[0] * src.res[1]))

def buffer_width(fc):
    return WIDTH.get(str(fc).split("_")[0], DEFAULT_W)

def paved_fraction(geom):
    """fraction of paved pixels inside geom, via a windowed mask read."""
    b = geom.bounds
    try:
        win = from_bounds(b[0], b[1], b[2], b[3], src.transform).round_offsets().round_lengths()
    except Exception:
        return np.nan
    r0, c0 = max(0, win.row_off), max(0, win.col_off)
    r1, c1 = min(paved.shape[0], win.row_off + win.height), min(paved.shape[1], win.col_off + win.width)
    if r1 <= r0 or c1 <= c0:
        return np.nan
    sub = paved[r0:r1, c0:c1]
    m = ~geometry_mask([geom], out_shape=sub.shape,
                       transform=rasterio.windows.transform(
                           rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0), src.transform))
    return float(sub[m].mean()) if m.any() else np.nan

def analyse(tag, path):
    if src is None:
        init()
    g = gpd.read_file(path).to_crs(src.crs)
    g = gpd.clip(g, shapely_box(*src.bounds)).explode(index_parts=False)  # NAIP coverage only
    g = g[g.geometry.length > 1].reset_index(drop=True)
    g["w"] = g["fclass"].map(buffer_width)
    g["support"] = [paved_fraction(row.geometry.buffer(row.w / 2 + TOL))
                    for row in g.itertuples()]
    g["supported"] = g["support"] >= SUPPORT_THR
    L = g.geometry.length

    # reverse: paved explained by any buffered way
    road_mask = rasterize(((row.geometry.buffer(row.w / 2 + TOL), 1) for row in g.itertuples()),
                          out_shape=paved.shape, transform=src.transform, fill=0, dtype="uint8")
    explained = float(paved[(paved == 1) & (road_mask == 1)].size / max(1, (paved == 1).sum()))

    by_class = {fc: dict(n=int(len(s)), km=round(float(s.geometry.length.sum() / 1000), 2),
                         supported=round(float(s["supported"].mean()), 3))
                for fc, s in g.groupby("fclass") if len(s) >= 5}
    res = dict(snapshot=tag, segments=int(len(g)), length_km=round(float(L.sum() / 1000), 2),
               support_threshold=SUPPORT_THR,
               supported_segments=round(float(g["supported"].mean()), 4),
               supported_length=round(float(L[g["supported"]].sum() / L.sum()), 4),
               median_support=round(float(g["support"].median()), 3),
               paved_explained=round(explained, 4), by_class=by_class)

    ext = rasterio.plot.plotting_extent(src)
    fig, ax = plt.subplots(1, 2, figsize=(19, 9))
    ax[0].imshow(np.where(paved == 1, 0.15, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1)
    g[g.supported].plot(ax=ax[0], color="#4c9f70", lw=0.7)
    g[~g.supported].plot(ax=ax[0], color="#d1483f", lw=1.2)
    ax[0].legend(handles=[Patch(color="#4c9f70", label=f"NAIP-supported ({res['supported_segments']:.0%})"),
                          Patch(color="#d1483f", label="unsupported"),
                          Patch(color="0.85", label="NAIP paved")], loc="lower center", ncol=3)
    ax[0].set_title(f"OSM {tag} ways vs NAIP paved — {res['supported_length']:.1%} of length supported")
    unexp = (paved == 1) & (road_mask == 0)
    ax[1].imshow(np.where(paved == 1, 0.15, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1)
    ax[1].imshow(np.where(unexp, 0.75, np.nan), extent=ext, cmap="autumn_r", vmin=0, vmax=1)
    ax[1].set_title(f"Paved not explained by OSM {tag} ({1 - res['paved_explained']:.1%} of paved area)\n"
                    "orange = candidate unmapped ways / parking / plazas")
    for a_ in ax:
        a_.set_xlim(ext[0], ext[1]); a_.set_ylim(ext[2], ext[3])
        a_.set_aspect("equal"); a_.ticklabel_format(style="plain")
    fig.suptitle(f"Roads: OSM {tag} vs NAIP paved surface", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"roads_{tag}.png"), dpi=130); plt.close(fig)
    return res

import rasterio.plot  # noqa: E402  (used above for plotting_extent)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 3:
        # region mode: <paved.tif> <out_dir> <tag=roads.geojson> [...]
        init(sys.argv[1], sys.argv[2])
        runs = dict(a.split("=", 1) for a in sys.argv[3:])
    else:
        runs = {y: os.path.join(ROOT, "data", f"osm_roads_{y}.geojson")
                for y in ("2019", "2026")}
    out = {tag: analyse(tag, path) for tag, path in runs.items()}
    json.dump(out, open(os.path.join(OUT, "roads_summary.json"), "w"), indent=2)
    print(json.dumps({y: {k: v for k, v in r.items() if k != "by_class"} for y, r in out.items()}, indent=2))
    print("done ->", OUT)

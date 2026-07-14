"""
Classical LiDAR object detection / segmentation for the UIUC campus tile.
Produces, from UIUC_campus_LiDAR_merged_2x2km.laz:
  1. Ground / DTM surface  -> results/dtm.tif, results/ground_dtm.png
  2. Building instances     -> results/buildings.geojson, results/buildings_detected.png
  3. Individual trees       -> results/trees.geojson, results/trees_detected.png
Runs on CPU in a few minutes. No training required.
"""
import os, json, numpy as np, laspy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, binary_closing, disk, remove_small_holes
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian
import rasterio
from rasterio.transform import from_origin
from rasterio import features
from rasterio.crs import CRS
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "UIUC_campus_LiDAR_merged_2x2km.laz")
OUT = os.path.join(ROOT, "outputs", "detection"); os.makedirs(OUT, exist_ok=True)
RES = 0.5  # metres / cell

h = laspy.open(SRC).header
xmin, ymin, xmax, ymax = h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]
nx = int(round((xmax - xmin) / RES)); ny = int(round((ymax - ymin) / RES))
transform = from_origin(xmin, ymax, RES, RES)          # north-up affine
albers = CRS.from_epsg(6350)
to_ll = Transformer.from_crs("EPSG:6350", "EPSG:4326", always_xy=True)

def rc(x, y):
    col = np.clip(((x - xmin) / RES).astype(np.int64), 0, nx - 1)
    row = np.clip(((ymax - y) / RES).astype(np.int64), 0, ny - 1)
    return row * nx + col

# ---------------------------------------------------------------- pass 1: rasterize
print(f"[1/5] rasterizing {h.point_count:,} pts -> {ny}x{nx} @ {RES} m")
dsm      = np.full(ny * nx, -np.inf, np.float32)   # max Z, all points  (surface)
grd      = np.full(ny * nx,  np.inf, np.float32)   # min Z, ground (cls 2)  (terrain)
bldg_cnt = np.zeros(ny * nx, np.int32)             # class 6 count
veg_cnt  = np.zeros(ny * nx, np.int32)             # class 5 (high veg) count
with laspy.open(SRC) as r:
    for pts in r.chunk_iterator(10_000_000):
        x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
        c = np.asarray(pts.classification)
        idx = rc(x, y)
        surf = (c != 7) & (c != 18)          # drop noise (Low Point / High Noise) from surface
        np.maximum.at(dsm, idx[surf], z[surf])
        g = c == 2
        if g.any(): np.minimum.at(grd, idx[g], z[g])
        b = c == 6
        if b.any(): np.add.at(bldg_cnt, idx[b], 1)
        v = c == 5
        if v.any(): np.add.at(veg_cnt, idx[v], 1)

dsm = dsm.reshape(ny, nx); grd = grd.reshape(ny, nx)
bldg_cnt = bldg_cnt.reshape(ny, nx); veg_cnt = veg_cnt.reshape(ny, nx)
dsm[~np.isfinite(dsm)] = np.nan
grd[~np.isfinite(grd)] = np.nan

# ---------------------------------------------------------------- ground / DTM
print("[2/5] building DTM (nearest-fill of ground minima)")
mask_nd = np.isnan(grd)
# fill terrain nodata (water, under buildings) with nearest ground value
_, (ir, ic) = ndi.distance_transform_edt(mask_nd, return_indices=True)
dtm = grd[ir, ic].astype(np.float32)
chm = np.where(np.isnan(dsm), 0, dsm) - dtm
chm = np.clip(chm, 0, None).astype(np.float32)

with rasterio.open(os.path.join(OUT, "dtm.tif"), "w", driver="GTiff",
                   height=ny, width=nx, count=1, dtype="float32",
                   crs=albers, transform=transform, nodata=np.nan) as dst:
    dst.write(dtm, 1)
with rasterio.open(os.path.join(OUT, "chm.tif"), "w", driver="GTiff",
                   height=ny, width=nx, count=1, dtype="float32",
                   crs=albers, transform=transform, nodata=0) as dst:
    dst.write(chm, 1)

ls = LightSource(azdeg=315, altdeg=45)
fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
ax.imshow(ls.hillshade(dtm, vert_exag=3, dx=RES, dy=RES), cmap="gray",
          extent=[xmin, xmax, ymin, ymax])
im = ax.imshow(dtm, cmap="terrain", alpha=0.5, extent=[xmin, xmax, ymin, ymax])
ax.set_title(f"Bare-earth DTM (ground class 2) — {RES} m")
plt.colorbar(im, ax=ax, shrink=.7, label="elevation NAVD88 (m)")
ax.ticklabel_format(style="plain"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "ground_dtm.png")); plt.close(fig)

# ---------------------------------------------------------------- building instances
print("[3/5] detecting building instances")
MIN_AREA_M2, MIN_HEIGHT_M = 20.0, 2.0
min_cells = int(MIN_AREA_M2 / (RES * RES))
bmask = bldg_cnt >= 2
bmask = binary_closing(bmask, disk(2))
bmask = remove_small_holes(bmask, area_threshold=min_cells)
bmask = remove_small_objects(bmask, min_size=min_cells)
lbl, n = ndi.label(bmask)

# drop low components (misclassified ground) using CHM
keep = np.zeros(n + 1, bool)
props = {p.label: p for p in regionprops(lbl)}
for lab, p in props.items():
    if np.nanmedian(chm[lbl == lab]) >= MIN_HEIGHT_M:
        keep[lab] = True
bmask_final = keep[lbl]
lbl2, nb = ndi.label(bmask_final)
print(f"      -> {nb} buildings")

feats, bboxes = [], []
for p in regionprops(lbl2):
    region = lbl2 == p.label
    hgt = float(np.nanmedian(chm[region]))
    area = float(p.area * RES * RES)
    minr, minc, maxr, maxc = p.bbox
    bx0, by1 = xmin + minc * RES, ymax - minr * RES
    bx1, by0 = xmin + maxc * RES, ymax - maxr * RES
    bboxes.append((bx0, by0, bx1, by1, p.label))
    feats.append(dict(label=int(p.label), area_m2=round(area, 1),
                      height_m=round(hgt, 1)))

# vectorize footprints -> lon/lat GeoJSON
geoms = []
for geom, val in features.shapes(lbl2.astype(np.int32), mask=bmask_final, transform=transform):
    if val == 0: continue
    poly = shape(geom).simplify(0.5)
    poly_ll = shp_transform(lambda xs, ys, z=None: to_ll.transform(xs, ys), poly)
    meta = next(f for f in feats if f["label"] == int(val))
    geoms.append(dict(type="Feature", properties={**meta, "class": "building"},
                      geometry=mapping(poly_ll)))
with open(os.path.join(OUT, "buildings.geojson"), "w") as f:
    json.dump(dict(type="FeatureCollection", crs={"type":"name",
              "properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}}, features=geoms), f)

rng = np.random.default_rng(0)
colors = np.vstack([[0, 0, 0], rng.uniform(.25, 1, (nb, 3))])
inst_rgb = colors[lbl2]
fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
ax.imshow(ls.hillshade(np.nan_to_num(dsm, nan=np.nanmin(dsm)), vert_exag=2, dx=RES, dy=RES),
          cmap="gray", extent=[xmin, xmax, ymin, ymax])
ax.imshow(np.dstack([inst_rgb, (lbl2 > 0) * 0.55]), extent=[xmin, xmax, ymin, ymax])
for bx0, by0, bx1, by1, lab in bboxes:
    ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                 fill=False, ec="red", lw=0.6))
ax.set_title(f"Building instance detection — {nb} buildings\n"
             f"(class-6 points -> morphology -> connected components; red = bbox)")
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.ticklabel_format(style="plain"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "buildings_detected.png")); plt.close(fig)

# ---------------------------------------------------------------- individual trees
print("[4/5] detecting individual trees (CHM local maxima + watershed crowns)")
bldg_dil = ndi.binary_dilation(bmask_final, disk(2))
TREE_MIN_H = 3.0
tree_region = (chm > TREE_MIN_H) & (veg_cnt >= 1) & (~bldg_dil)
chm_s = gaussian(chm, sigma=1.0, preserve_range=True)
chm_s[~tree_region] = 0
tops = peak_local_max(chm_s, min_distance=6, threshold_abs=TREE_MIN_H, labels=tree_region)
markers = np.zeros_like(chm_s, np.int32)
for i, (rr, cc) in enumerate(tops, 1): markers[rr, cc] = i
crowns = watershed(-chm_s, markers, mask=tree_region)
nt = len(tops)
print(f"      -> {nt} trees")

crown_area = ndi.sum(np.ones_like(crowns), crowns, index=np.arange(1, nt + 1)) * RES * RES
tfeats = []
for i, (rr, cc) in enumerate(tops):
    lon, lat = to_ll.transform(xmin + cc * RES, ymax - rr * RES)
    tfeats.append(dict(type="Feature",
        properties=dict(tree_id=i + 1, height_m=round(float(chm[rr, cc]), 1),
                        crown_m2=round(float(crown_area[i]), 1), **{"class": "tree"}),
        geometry=dict(type="Point", coordinates=[round(lon, 8), round(lat, 8)])))
with open(os.path.join(OUT, "trees.geojson"), "w") as f:
    json.dump(dict(type="FeatureCollection", crs={"type":"name",
              "properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}}, features=tfeats), f)

fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
im = ax.imshow(np.where(tree_region, chm, np.nan), cmap="YlGn", vmax=25,
               extent=[xmin, xmax, ymin, ymax])
ty = ymax - tops[:, 0] * RES; tx = xmin + tops[:, 1] * RES
ax.scatter(tx, ty, s=2, c="darkred", marker="^")
ax.set_title(f"Individual tree detection — {nt} trees\n"
             f"(canopy height model, local maxima ≥{TREE_MIN_H:.0f} m, min spacing 3 m)")
plt.colorbar(im, ax=ax, shrink=.7, label="canopy height (m)")
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.ticklabel_format(style="plain"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "trees_detected.png")); plt.close(fig)

# ---------------------------------------------------------------- summary
print("[5/5] summary")
summary = dict(
    resolution_m=RES, grid=[ny, nx],
    buildings=dict(count=nb,
        total_footprint_m2=round(sum(f["area_m2"] for f in feats), 1),
        median_height_m=round(float(np.median([f["height_m"] for f in feats])), 1),
        tallest_m=round(float(max(f["height_m"] for f in feats)), 1)),
    trees=dict(count=nt,
        median_height_m=round(float(np.median([f["properties"]["height_m"] for f in tfeats])), 1),
        tallest_m=round(float(max(f["properties"]["height_m"] for f in tfeats)), 1),
        median_crown_m2=round(float(np.median([f["properties"]["crown_m2"] for f in tfeats])), 1)))
with open(os.path.join(OUT, "detection_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print("\nOutputs in", OUT)

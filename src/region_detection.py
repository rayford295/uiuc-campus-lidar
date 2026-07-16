"""Region-generic LiDAR detection for minimally-classified point clouds.

classical_detection.py leans on the ASPRS building (6) and high-vegetation (5)
classes that the UIUC QL1 delivery ships with. Many 3DEP deliveries (e.g. the
Colorado Springs tile) are only ground/non-ground classified, so this variant
replaces the class evidence with two sensor-derived signals while keeping the
same grid, morphology and thresholds — cross-region numbers stay comparable:

  vegetation  NAIP NDVI (warped onto the LiDAR grid)  OR  the multi-return
              echo fraction per cell (canopies produce intermediate returns;
              roofs are single-return)
  buildings   elevated (CHM >= 2 m), NOT vegetation, morphology-cleaned,
              components >= 20 m2 with median CHM >= 2 m
  trees       CHM > 3 m AND vegetation, local maxima + watershed crowns

Outputs use the same filenames/schema as classical_detection.py
(buildings.geojson, trees.geojson, dtm/chm.tif, PNGs, detection_summary.json)
so every downstream comparison script works unchanged.

Usage:  python src/region_detection.py <lidar.las|laz> <naip.tif> <out_dir>
"""
import json
import os
import sys

import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource
from pyproj import Transformer
from rasterio import features
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from scipy import ndimage as ndi
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.morphology import (binary_closing, disk, remove_small_holes,
                                remove_small_objects)
from skimage.segmentation import watershed

SRC, NAIP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
RES = 0.5
MIN_AREA_M2, MIN_HEIGHT_M, TREE_MIN_H = 20.0, 2.0, 3.0
NDVI_VEG, ECHO_VEG = 0.25, 0.35          # either signal marks vegetation

h = laspy.open(SRC).header
xmin, ymin, xmax, ymax = h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]
nx = int(round((xmax - xmin) / RES))
ny = int(round((ymax - ymin) / RES))
transform = from_origin(xmin, ymax, RES, RES)
albers = CRS.from_epsg(6350)
to_ll = Transformer.from_crs("EPSG:6350", "EPSG:4326", always_xy=True)


def rc(x, y):
    col = np.clip(((x - xmin) / RES).astype(np.int64), 0, nx - 1)
    row = np.clip(((ymax - y) / RES).astype(np.int64), 0, ny - 1)
    return row * nx + col


# ---------------------------------------------------------------- rasterize
print(f"[1/5] rasterizing {h.point_count:,} pts -> {ny}x{nx} @ {RES} m")
dsm = np.full(ny * nx, -np.inf, np.float32)
grd = np.full(ny * nx, np.inf, np.float32)
echo_multi = np.zeros(ny * nx, np.int32)
echo_all = np.zeros(ny * nx, np.int32)
with laspy.open(SRC) as r:
    for pts in r.chunk_iterator(10_000_000):
        x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
        c = np.asarray(pts.classification)
        nr = np.asarray(pts.number_of_returns)
        idx = rc(x, y)
        surf = (c != 7) & (c != 18)
        np.maximum.at(dsm, idx[surf], z[surf])
        g = c == 2
        if g.any():
            np.minimum.at(grd, idx[g], z[g])
        np.add.at(echo_all, idx, 1)
        np.add.at(echo_multi, idx[nr > 1], 1)

dsm = dsm.reshape(ny, nx)
grd = grd.reshape(ny, nx)
echo_frac = (echo_multi / np.maximum(echo_all, 1)).reshape(ny, nx)
dsm[~np.isfinite(dsm)] = np.nan
grd[~np.isfinite(grd)] = np.nan

# ---------------------------------------------------------------- DTM / CHM
print("[2/5] building DTM (nearest-fill of ground minima)")
mask_nd = np.isnan(grd)
_, (ir, ic) = ndi.distance_transform_edt(mask_nd, return_indices=True)
dtm = grd[ir, ic].astype(np.float32)
chm = np.clip(np.where(np.isnan(dsm), 0, dsm) - dtm, 0, None).astype(np.float32)

for name, arr, nodata in (("dtm", dtm, np.nan), ("chm", chm, 0)):
    with rasterio.open(os.path.join(OUT, f"{name}.tif"), "w", driver="GTiff",
                       height=ny, width=nx, count=1, dtype="float32",
                       crs=albers, transform=transform, nodata=nodata) as dst:
        dst.write(arr, 1)

# ---------------------------------------------------------------- NDVI on grid
print("[3/5] warping NAIP NDVI onto the LiDAR grid")
with rasterio.open(NAIP) as naip:
    with WarpedVRT(naip, crs=albers, transform=transform, width=nx, height=ny,
                   resampling=Resampling.bilinear) as vrt:
        red = vrt.read(1).astype("float32")
        nir = vrt.read(4).astype("float32")
ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
veg = (ndvi >= NDVI_VEG) | (echo_frac >= ECHO_VEG)

# ---------------------------------------------------------------- buildings
print("[4/5] detecting buildings (elevated, non-vegetation)")
min_cells = int(MIN_AREA_M2 / (RES * RES))
bmask = (chm >= MIN_HEIGHT_M) & ~veg
bmask = binary_closing(bmask, disk(2))
bmask = remove_small_holes(bmask, area_threshold=min_cells)
bmask = remove_small_objects(bmask, min_size=min_cells)
lbl, n = ndi.label(bmask)
keep = np.zeros(n + 1, bool)
for p in regionprops(lbl):
    if np.nanmedian(chm[lbl == p.label]) >= MIN_HEIGHT_M:
        keep[p.label] = True
bmask_final = keep[lbl]
lbl2, nb = ndi.label(bmask_final)
print(f"      -> {nb} buildings")

ls = LightSource(azdeg=315, altdeg=45)
feats = []
for p in regionprops(lbl2):
    region = lbl2 == p.label
    feats.append(dict(label=int(p.label),
                      area_m2=round(float(p.area * RES * RES), 1),
                      height_m=round(float(np.nanmedian(chm[region])), 1)))
geoms = []
for geom, val in features.shapes(lbl2.astype(np.int32), mask=bmask_final,
                                 transform=transform):
    if val == 0:
        continue
    poly = shape(geom).simplify(0.5)
    poly_ll = shp_transform(lambda xs, ys, z=None: to_ll.transform(xs, ys), poly)
    meta = next(f for f in feats if f["label"] == int(val))
    geoms.append(dict(type="Feature", properties={**meta, "class": "building"},
                      geometry=mapping(poly_ll)))
with open(os.path.join(OUT, "buildings.geojson"), "w") as f:
    json.dump(dict(type="FeatureCollection", crs={"type": "name", "properties":
              {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, features=geoms), f)

rng = np.random.default_rng(0)
colors = np.vstack([[0, 0, 0], rng.uniform(.25, 1, (max(nb, 1), 3))])
fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
ax.imshow(ls.hillshade(np.nan_to_num(dsm, nan=np.nanmin(dsm)), vert_exag=2,
                       dx=RES, dy=RES), cmap="gray",
          extent=[xmin, xmax, ymin, ymax])
ax.imshow(np.dstack([colors[lbl2], (lbl2 > 0) * 0.55]),
          extent=[xmin, xmax, ymin, ymax])
ax.set_title(f"Building detection — {nb} buildings\n"
             "(CHM >= 2 m, non-vegetation by NDVI/echo, morphology)")
ax.ticklabel_format(style="plain")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "buildings_detected.png"))
plt.close(fig)

# ---------------------------------------------------------------- trees
print("[5/5] detecting trees")
bldg_dil = ndi.binary_dilation(bmask_final, disk(2))
tree_region = (chm > TREE_MIN_H) & veg & (~bldg_dil)
chm_s = gaussian(chm, sigma=1.0, preserve_range=True)
chm_s[~tree_region] = 0
tops = peak_local_max(chm_s, min_distance=6, threshold_abs=TREE_MIN_H,
                      labels=tree_region)
markers = np.zeros_like(chm_s, np.int32)
for i, (rr, cc) in enumerate(tops, 1):
    markers[rr, cc] = i
crowns = watershed(-chm_s, markers, mask=tree_region)
nt = len(tops)
print(f"      -> {nt} trees")

crown_area = ndi.sum(np.ones_like(crowns), crowns,
                     index=np.arange(1, nt + 1)) * RES * RES
tfeats = []
for i, (rr, cc) in enumerate(tops):
    lon, lat = to_ll.transform(xmin + cc * RES, ymax - rr * RES)
    tfeats.append(dict(type="Feature", properties=dict(
        tree_id=i + 1, height_m=round(float(chm[rr, cc]), 1),
        crown_m2=round(float(crown_area[i]), 1), **{"class": "tree"}),
        geometry=dict(type="Point", coordinates=[round(lon, 8), round(lat, 8)])))
with open(os.path.join(OUT, "trees.geojson"), "w") as f:
    json.dump(dict(type="FeatureCollection", crs={"type": "name", "properties":
              {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, features=tfeats), f)

fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
im = ax.imshow(np.where(tree_region, chm, np.nan), cmap="YlGn", vmax=25,
               extent=[xmin, xmax, ymin, ymax])
if nt:
    ax.scatter(xmin + tops[:, 1] * RES, ymax - tops[:, 0] * RES,
               s=2, c="darkred", marker="^")
ax.set_title(f"Tree detection — {nt} trees (CHM maxima, NDVI/echo vegetation)")
plt.colorbar(im, ax=ax, shrink=.7, label="canopy height (m)")
ax.ticklabel_format(style="plain")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "trees_detected.png"))
plt.close(fig)

summary = dict(
    resolution_m=RES, grid=[ny, nx], vegetation_source="NDVI+echo",
    buildings=dict(count=nb,
                   total_footprint_m2=round(sum(f["area_m2"] for f in feats), 1),
                   median_height_m=round(float(np.median(
                       [f["height_m"] for f in feats])), 1) if feats else None),
    trees=dict(count=nt))
with open(os.path.join(OUT, "detection_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

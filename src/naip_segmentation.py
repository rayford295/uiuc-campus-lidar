"""
NAIP land-cover + building/road (paved) segmentation — the optical RS source.

NAIP is 4-band (R,G,B,NIR) ~0.7 m imagery. Optical imagery has no height, so buildings and
roads are near-identical spectrally and CANNOT be separated by NAIP alone (in a dense scene
all impervious pixels form one connected blob — verified). We therefore fuse:

    vegetation  = NDVI (NIR) OR ExG (visible)         # robust, NAIP-only
    impervious  = valid & not vegetation & not water  # built-up extent, NAIP-only
    building    = impervious ∩ LiDAR footprint        # height from LiDAR resolves buildings
    paved       = impervious − building − vegetation   # roads + parking + sidewalks (ground)

Using LiDAR to *remove* buildings is not circular for the road comparison (roads are
evaluated OSM-vs-NAIP; LiDAR is not the road reference). "paved" is a superset of roads
(includes parking/plazas) — true road centrelines come from the OSM-vs-NAIP step / a
road model. NAIP is NOT trained on LiDAR, so its impervious extent stays independent.

Usage:  python src/naip_segmentation.py <NAIP_image.tif>
"""
import os, sys, json, numpy as np, rasterio, geopandas as gpd
from rasterio.windows import from_bounds
from rasterio.features import rasterize
from rasterio.crs import CRS
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
    sys.exit("usage: python src/naip_segmentation.py <NAIP_image.tif> "
             "[lidar_buildings.geojson] [out_dir]")
NAIP = sys.argv[1]
LIDAR_BLDG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    ROOT, "results", "detection", "buildings.geojson")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "results", "naip")
os.makedirs(OUT, exist_ok=True)
if len(sys.argv) > 2:      # region mode: the raster is already clipped to the tile
    with rasterio.open(NAIP) as _ds:
        TILE = tuple(_ds.bounds)
else:
    TILE = (655000, 1925000, 657000, 1927000)  # EPSG:6350, campus LiDAR tile

with rasterio.open(NAIP) as ds:
    win = from_bounds(*TILE, transform=ds.transform)
    img = ds.read(window=win).astype(np.float32)     # (4,H,W) R,G,B,NIR
    wt = ds.window_transform(win); res = ds.res
R, G, B, NIR = img
H, W = R.shape; pxa = res[0]*res[1]
print(f"NAIP tile {W}x{H} px @ {res[0]:.2f} m, EPSG:6350")

valid = img.sum(0) > 0
ndvi = (NIR - R) / (NIR + R + 1e-6)
exg = 2*G - R - B                                    # visible green (weak-NIR fallback)
bright = (R + G + B) / 3
veg = valid & ((ndvi > 0.05) | (exg > 25))
water = valid & ~veg & (bright < 35)
imperv = valid & ~veg & ~water
imperv = remove_small_objects(remove_small_holes(imperv, 60), 40)

# LiDAR footprints -> NAIP grid resolve buildings within impervious
bl = gpd.read_file(LIDAR_BLDG).to_crs(6350); bl["geometry"] = bl.buffer(0)
lidar_bldg = rasterize(((g, 1) for g in bl.geometry), out_shape=(H, W),
                       transform=wt, fill=0, dtype="uint8").astype(bool)
building = imperv & lidar_bldg
paved = imperv & ~ndi.binary_dilation(building, disk(2)) & ~veg   # roads + parking + sidewalks
paved = remove_small_objects(paved, 60)

# class raster: 1 veg, 2 water/shadow, 3 building, 4 paved(non-building impervious)
cls = np.zeros((H, W), np.uint8)
cls[veg] = 1; cls[water] = 2; cls[paved] = 4; cls[building] = 3
for name, arr in [("building", building), ("paved", paved), ("vegetation", veg)]:
    with rasterio.open(os.path.join(OUT, f"naip_{name}.tif"), "w", driver="GTiff", height=H,
                       width=W, count=1, dtype="uint8", crs=CRS.from_epsg(6350),
                       transform=wt, nodata=0, compress="lzw") as d:
        d.write(arr.astype(np.uint8), 1)
with rasterio.open(os.path.join(OUT, "naip_landcover.tif"), "w", driver="GTiff", height=H,
                   width=W, count=1, dtype="uint8", crs=CRS.from_epsg(6350),
                   transform=wt, nodata=255, compress="lzw") as d:
    d.write(cls, 1)

summary = dict(resolution_m=round(res[0], 3),
    vegetation_pct=round(100*float(veg.sum())/valid.sum(), 1),
    impervious_pct=round(100*float(imperv.sum())/valid.sum(), 1),
    building_m2=round(float(building.sum()*pxa)),
    paved_m2=round(float(paved.sum()*pxa)),
    vegetation_m2=round(float(veg.sum()*pxa)),
    note="building = NAIP impervious ∩ LiDAR footprint; paved = roads+parking+sidewalks")
json.dump(summary, open(os.path.join(OUT, "naip_summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))

ext = [TILE[0], TILE[2], TILE[1], TILE[3]]
CM = ListedColormap(["#e8e4d8", "#4c9f4c", "#3b6fb0", "#d1483f", "#555555"])
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
ax[0].imshow(np.dstack([R, G, B])/255.0, extent=ext, origin="upper"); ax[0].set_title("NAIP true colour")
nd = ax[1].imshow(np.where(valid, ndvi, np.nan), extent=ext, origin="upper", cmap="RdYlGn", vmin=-.4, vmax=.4)
ax[1].set_title("NDVI"); plt.colorbar(nd, ax=ax[1], shrink=.7)
ax[2].imshow(cls, cmap=CM, vmin=0, vmax=4, extent=ext, origin="upper", interpolation="nearest")
ax[2].set_title("NAIP land cover (LiDAR-fused)")
ax[2].legend(handles=[Patch(color="#d1483f", label="building (∩ LiDAR)"), Patch(color="#555555", label="paved: road/parking"),
                      Patch(color="#4c9f4c", label="vegetation"), Patch(color="#3b6fb0", label="water/shadow"),
                      Patch(color="#e8e4d8", label="other")], loc="lower center", ncol=3, fontsize=8)
for a in ax: a.ticklabel_format(style="plain")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "naip_segmentation.png"), dpi=130); plt.close(fig)
print("done ->", OUT)

"""Download & stage every external input of the pipeline (idempotent).

The repo ships only small, git-friendly files (campus OSM subsets, metadata).
Everything heavy lives on public storage and is fetched on demand:

  LiDAR   4x1 km USGS 3DEP QL1 tiles     I-GUIDE dataset storage -> merged .laz
  NAIP    4-band aerial image (~0.7 m)   GitHub release `campus-rs-2019`
  OSM     statewide major-roads extract  GitHub release `osm-il-2019`
  Census  county area/population/bounds  census.gov static files

Everything lands under the repo root in gitignored locations, matching the
paths the src/ scripts expect, so this works identically on a laptop and on
the I-GUIDE JupyterHub.

Usage:  python src/prepare_data.py [lidar] [naip] [statewide]
        (no args = all three)
"""
import os
import sys
import zipfile
from urllib.request import urlretrieve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
STATEWIDE = os.path.join(DATA, "statewide")

RELEASE = "https://github.com/rayford295/vgi-spatial-bias/releases/download"
URLS = {
    "lidar": "https://storage.i-guide.io/datasets/9c1842f8-1d74-4b88-8ad3-cf0a0cb47a86/uiuc_campus_lidar.zip",
    "naip": f"{RELEASE}/campus-rs-2019/NAIP_image.tif",
    "roads": f"{RELEASE}/osm-il-2019/OSM_2019_IL_Major_Roads.zip",
    "gazetteer": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2019_Gazetteer/2019_gaz_counties_17.txt",
    "population": "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv",
    "boundaries": "https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_county_20m.zip",
}

URLS["cs_lidar"] = f"{RELEASE}/colorado-springs-2019/cs_lidar_2km.laz"
URLS["cs_naip"] = f"{RELEASE}/colorado-springs-2019/colorado_springs_NAIP_clipped_6350.tif"

LAZ = os.path.join(ROOT, "UIUC_campus_LiDAR_merged_2x2km.laz")
NAIP = os.path.join(DATA, "NAIP_image.tif")
ROADS_SHP = os.path.join(STATEWIDE, "OSM_2019_Major_Roads",
                         "gis_osm_roads_2019_IL_Major_Roads.shp")
CS_DATA = os.path.join(ROOT, "regions", "colorado_springs", "data")


def fetch(url, dest):
    if os.path.exists(dest):
        print(f"  already present: {os.path.relpath(dest, ROOT)}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url.split('/')[-1]} ...")
    urlretrieve(url, dest + ".part")
    os.rename(dest + ".part", dest)
    print(f"  -> {os.path.relpath(dest, ROOT)} ({os.path.getsize(dest)/1e6:.0f} MB)")
    return dest


def prepare_lidar():
    """I-GUIDE zip holds 4 x 1 km tiles; the scripts expect one merged cloud."""
    print("[lidar]")
    if os.path.exists(LAZ):
        print(f"  already present: {os.path.basename(LAZ)}")
        return
    import glob

    import laspy

    workdir = os.path.join(ROOT, "data", "lidar_tiles")
    z = fetch(URLS["lidar"], os.path.join(workdir, "uiuc_campus_lidar.zip"))
    with zipfile.ZipFile(z) as zf:
        zf.extractall(workdir)
    tiles = sorted(glob.glob(os.path.join(workdir, "**", "*.la[sz]"), recursive=True))
    merged = [p for p in tiles if "merged" in os.path.basename(p).lower()]
    if merged:  # dataset already ships the merged cloud
        os.replace(merged[0], LAZ)
    else:
        with laspy.open(tiles[0]) as r0:
            header = r0.header
        total = 0
        with laspy.open(LAZ, mode="w", header=header) as w:
            for t in tiles:
                with laspy.open(t) as r:
                    for pts in r.chunk_iterator(5_000_000):
                        w.write_points(pts)
                        total += len(pts)
        print(f"  merged {len(tiles)} tiles ({total:,} pts)")
    print(f"  -> {os.path.basename(LAZ)}")


def prepare_naip():
    print("[naip]")
    fetch(URLS["naip"], NAIP)


def prepare_statewide():
    print("[statewide]")
    z = fetch(URLS["roads"], os.path.join(STATEWIDE, "OSM_2019_IL_Major_Roads.zip"))
    if not os.path.exists(ROADS_SHP):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(STATEWIDE)
        print(f"  -> {os.path.relpath(ROADS_SHP, ROOT)}")
    fetch(URLS["gazetteer"], os.path.join(STATEWIDE, "gaz_counties_17.txt"))
    fetch(URLS["population"], os.path.join(STATEWIDE, "co-est2019.csv"))
    zb = fetch(URLS["boundaries"], os.path.join(STATEWIDE, "cb_county.zip"))
    cb = os.path.join(STATEWIDE, "cb_county")
    if not os.path.exists(os.path.join(cb, "cb_2019_us_county_20m.shp")):
        with zipfile.ZipFile(zb) as zf:
            zf.extractall(cb)


def prepare_colorado():
    """Second study region (LiDAR + clipped NAIP; OSM clips are committed)."""
    print("[colorado]")
    fetch(URLS["cs_lidar"], os.path.join(CS_DATA, "cs_lidar_2km.laz"))
    fetch(URLS["cs_naip"], os.path.join(CS_DATA, "naip_cs_6350.tif"))


STEPS = {"lidar": prepare_lidar, "naip": prepare_naip,
         "statewide": prepare_statewide, "colorado": prepare_colorado}

if __name__ == "__main__":
    for step in (sys.argv[1:] or STEPS):
        STEPS[step]()
    print("data ready")

"""
Fetch current OSM building=* polygons for the campus bbox via the Overpass API.

The 2019 snapshot (data/osm_buildings_2019.geojson) is the temporally-matched layer for
the bias study; the current snapshot fetched here serves as VALIDATION — if today's OSM
has filled the 2019 gaps, those gaps were real omissions (community under-mapping), not
buildings that didn't exist yet.

Usage:  python src/fetch_osm_current.py [out.geojson]   (default: data/osm_buildings_current.geojson)
"""
import os, sys, json, time, urllib.request
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "osm_buildings_current.geojson")
S, W, N, E = 40.0990944, -88.2402753, 40.1183436, -88.2147506   # campus tile, WGS84

query = f"""[out:json][timeout:120];
(way["building"]({S},{W},{N},{E}); relation["building"]({S},{W},{N},{E}););
out geom;"""
req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                             data=query.encode(), headers={"User-Agent": "vgi-spatial-bias/1.0"})
data = json.loads(urllib.request.urlopen(req, timeout=180).read())

feats = []
for el in data["elements"]:
    tags = el.get("tags", {})
    if el["type"] == "way" and el.get("geometry"):
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) >= 4 and coords[0] == coords[-1]:
            feats.append(dict(osm_id=el["id"], fclass="building",
                              type=tags.get("building", ""), geometry=Polygon(coords)))
    elif el["type"] == "relation" and el.get("members"):
        outers, inners = [], []
        for m in el["members"]:
            if m["type"] != "way" or not m.get("geometry"):
                continue
            ring = [(p["lon"], p["lat"]) for p in m["geometry"]]
            (outers if m.get("role") != "inner" else inners).append(ring)
        polys = list(polygonize([Polygon(r).exterior for r in outers if len(r) >= 4]))
        if polys:
            geom = unary_union(polys)
            for r in inners:
                if len(r) >= 4:
                    geom = geom.difference(Polygon(r))
            if not geom.is_empty:
                feats.append(dict(osm_id=el["id"], fclass="building",
                                  type=tags.get("building", ""), geometry=geom))

gdf = gpd.GeoDataFrame(feats, crs=4326)
gdf["geometry"] = gdf.geometry.buffer(0)
gdf = gdf[~gdf.geometry.is_empty]
gdf["retrieved"] = time.strftime("%Y-%m-%d")
gdf.to_file(OUT, driver="GeoJSON")
print(f"{len(gdf)} buildings -> {OUT}")

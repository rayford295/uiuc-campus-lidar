"""Campus major-roads clip (roads_clip_categories) — the vetted OSM 2019 subset.

The clip is the ArcGIS-filtered "major roads" extract (primary/secondary/
tertiary/residential, county-joined) cut to the 2x2 km campus tile —
the campus counterpart of the statewide file behind src/statewide_bias.py.

Two questions:
  1. Geometry: does every major way have NAIP pavement evidence?
     (reuses road_comparison.analyse — same buffers/thresholds as the
      all-class run, so numbers are directly comparable)
  2. Attributes: how complete are maxspeed/surface/name tags on campus vs
     statewide? (links the campus pilot to the statewide bias gradient)

Usage: python src/major_roads_analysis.py
Writes results/comparison/roads/roads_2019_major.png + major_roads_summary.json
"""
import json
import os

import geopandas as gpd
import pandas as pd

from road_comparison import OUT, ROOT, analyse

CLIP = os.path.join(ROOT, "data", "osm_roads_2019_major.geojson")

# statewide county means for context (results/statewide/county_metrics.csv)
STATEWIDE = {"pct_maxspeed": 1.44, "pct_surface": 1.72, "pct_name": 73.9}

res = analyse("2019_major", CLIP)

g = gpd.read_file(CLIP)
attrs = {
    "pct_maxspeed": float((pd.to_numeric(g.maxspeed, errors="coerce").fillna(0) > 0).mean()) * 100,
    "pct_surface": float((g.surface.notna() & (g.surface.astype(str).str.strip() != "")).mean()) * 100,
    "pct_name": float((g.loc_name.notna() & (g.loc_name.astype(str).str.strip() != "")).mean()) * 100,
}
edit_year = pd.to_datetime(g.lastchange).dt.year
res["attributes"] = {k: round(v, 1) for k, v in attrs.items()}
res["attributes_statewide_county_mean"] = STATEWIDE
res["median_edit_year"] = int(edit_year.median())
res["pct_edited_2017plus"] = round(float((edit_year >= 2017).mean()) * 100, 1)

json.dump(res, open(os.path.join(OUT, "major_roads_summary.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "by_class"}, indent=2))
print("by_class:", json.dumps(res["by_class"], indent=1))

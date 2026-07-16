"""Rule-based geometry proposals for detected OSM building omissions.

Turns each LiDAR-detected gap footprint into an OSM-style building polygon:

  fill ratio >= 0.8  ->  minimum rotated rectangle (most small buildings)
  otherwise          ->  rotate to principal axis, simplify, orthogonalize the
                         ring (every edge snapped to 0/90 deg), rotate back
  failure            ->  simplified convex hull, flagged regularized=false

Also builds the evaluation scaffolding shared by every correction approach:
each gap is matched (IoU >= 0.3, same rule as temporal_validation.py) against
the OSM 2026 snapshot — community-drawn polygons are the geometry ground truth
for gaps the community filled, and the filled/still-unmapped label is what the
acceptance scorers try to predict.

Outputs (results/correction/):
  proposals_rule.geojson   one proposal per gap: building=yes, height, flags
  community_truth.geojson  the best-matching 2026 community polygon per filled gap
  gaps_labeled.geojson     gap footprints + filled label + east/west split flag

Usage:  python src/propose_geometry.py
"""
import json
import math
import os
import warnings

import geopandas as gpd
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "correction")
os.makedirs(OUT, exist_ok=True)
CRS, IOU_THR = 6350, 0.3
SPLIT_X = 656000.0          # tile midline: west = train, east = eval (same as DGCNN)
FILL_THR, SIMPLIFY_TOL = 0.8, 1.5


def largest_part(geom):
    geom = geom.buffer(0)
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom


def orthogonalize(ring_coords):
    """Walk the ring snapping every edge to the nearer of 0/90 degrees."""
    out = [ring_coords[0]]
    for x, y in ring_coords[1:]:
        px, py = out[-1]
        if abs(x - px) >= abs(y - py):
            out.append((x, py))
        else:
            out.append((px, y))
    out[-1] = out[0]
    return Polygon(out).buffer(0)


def regularize(geom):
    """Return (osm-style polygon, regularized flag)."""
    geom = largest_part(geom)
    with warnings.catch_warnings():
        # degenerate slivers make oriented_envelope emit RuntimeWarnings;
        # they fall through to the hull fallback below
        warnings.simplefilter("ignore", RuntimeWarning)
        mrr = geom.minimum_rotated_rectangle
    if mrr.is_empty or not math.isfinite(mrr.area) or mrr.area <= 0:
        return geom.convex_hull.simplify(1.0), False
    if geom.area / mrr.area >= FILL_THR:
        return mrr, True
    # principal angle = direction of the MRR's longest edge
    pts = list(mrr.exterior.coords)
    (x0, y0), (x1, y1) = max(((pts[i], pts[i + 1]) for i in range(4)),
                             key=lambda ab: (ab[1][0] - ab[0][0]) ** 2 +
                                            (ab[1][1] - ab[0][1]) ** 2)
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
    origin = (geom.centroid.x, geom.centroid.y)
    r = affinity.rotate(geom, -ang, origin=origin)
    poly = orthogonalize(list(r.simplify(SIMPLIFY_TOL).exterior.coords))
    poly = largest_part(poly) if not poly.is_empty else poly
    ok = (not poly.is_empty and poly.is_valid and
          0.5 <= poly.area / geom.area <= 2.0)
    if not ok:
        return geom.convex_hull.simplify(1.0), False
    return affinity.rotate(poly, ang, origin=origin), True


def best_match(gaps, ref):
    """Best-IoU 2026 polygon per gap (index and IoU; -1 if no intersection)."""
    pairs = gpd.sjoin(gaps[["geometry"]], ref[["geometry"]],
                      predicate="intersects", how="inner")
    best_iou = np.zeros(len(gaps))
    best_idx = np.full(len(gaps), -1)
    if pairs.empty:
        return best_iou, best_idx
    inter = pairs.geometry.values.intersection(
        ref.geometry.values[pairs.index_right.values]).area
    iou = inter / (gaps.area.values[pairs.index.values] +
                   ref.area.values[pairs.index_right.values] - inter)
    df = pairs.assign(iou=iou).reset_index()
    top = df.sort_values("iou").groupby("index").last()
    best_iou[top.index.values] = top["iou"].values
    best_idx[top.index.values] = top["index_right"].values
    return best_iou, best_idx


def main():
    gaps = gpd.read_file(os.path.join(ROOT, "results", "comparison",
                                      "omissions.geojson")).to_crs(CRS)
    gaps["geometry"] = gaps.geometry.buffer(0)
    gaps = gaps.reset_index(drop=True)
    gaps["gap_id"] = gaps.index

    o26 = gpd.read_file(os.path.join(ROOT, "data",
                                     "osm_buildings_2026.geojson")).to_crs(CRS)
    o26["geometry"] = o26.geometry.buffer(0)
    o26 = o26[o26.area >= 5.0].reset_index(drop=True)

    iou, idx = best_match(gaps, o26)
    gaps["filled"] = iou >= IOU_THR
    gaps["truth_iou"] = iou.round(3)
    gaps["east"] = gaps.geometry.centroid.x >= SPLIT_X

    props, flags = zip(*(regularize(g) for g in gaps.geometry))
    proposals = gpd.GeoDataFrame(
        {"gap_id": gaps.gap_id, "building": "yes",
         "height": gaps.height_m.round(1),
         "source": "USGS 3DEP LiDAR + NAIP consensus (vgi-spatial-bias)",
         "regularized": list(flags), "filled": gaps.filled, "east": gaps.east},
        geometry=list(props), crs=CRS)

    truth = o26.loc[idx[gaps.filled.values], ["geometry"]].reset_index(drop=True)
    truth["gap_id"] = gaps.gap_id[gaps.filled.values].values

    proposals.to_crs(4326).to_file(os.path.join(OUT, "proposals_rule.geojson"),
                                   driver="GeoJSON")
    truth.to_crs(4326).to_file(os.path.join(OUT, "community_truth.geojson"),
                               driver="GeoJSON")
    gaps.to_crs(4326).to_file(os.path.join(OUT, "gaps_labeled.geojson"),
                              driver="GeoJSON")

    summary = dict(gaps=int(len(gaps)), filled=int(gaps.filled.sum()),
                   still_unmapped=int((~gaps.filled).sum()),
                   regularized=int(sum(flags)),
                   hull_fallbacks=int(len(flags) - sum(flags)),
                   east_eval_filled=int((gaps.filled & gaps.east).sum()))
    json.dump(summary, open(os.path.join(OUT, "proposals_rule_summary.json"),
                            "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

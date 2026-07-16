"""Approach B — learned geometry proposals (small U-Net on NAIP + CHM).

Instead of hand-written regularization rules, learn what community-drawn
buildings look like: a 5-channel U-Net (NAIP R,G,B,NIR resampled to the CHM's
0.5 m grid + canopy-height model) is trained to segment OSM-2026 building
pixels on 64 m patches centered on the west-half filled gaps, then run on
every gap. The predicted component over the gap is polygonized and passed
through the same regularization exit as the rule pipeline, so A/B/C differ
only in where the geometry comes from.

Training labels are the 2026 community polygons — the model learns the
*mapping community's* notion of a building footprint, not LiDAR's. West half
trains (with flip/rotation augmentation), east half is never seen.

Outputs (results/correction/):
  proposals_learned.geojson  per-gap proposal + mask-probability confidence
  unet_metrics.json          training curve + east-half pixel IoU
  unet_weights.pt

Usage:  python src/propose_learned.py [epochs]
"""
import json
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.vrt import WarpedVRT
from shapely.geometry import shape as shp_shape

from propose_geometry import largest_part, regularize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGION = sys.argv[1] if len(sys.argv) > 1 else "uiuc_campus"
DATA = os.path.join(ROOT, "data", REGION)
RES = os.path.join(ROOT, "results", REGION)
OUT = os.path.join(RES, "correction")
# fixed west/east split line per region (tile midline, same as the DGCNN split)
SPLIT_X_BY_REGION = {"uiuc_campus": 656000.0, "colorado_springs": -752834.5}
CRS, PATCH = 6350, 128                    # 128 px @ 0.5 m = 64 m patches
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


def load_stack():
    """NAIP (RGBN) warped onto the CHM grid + CHM -> (5, H, W) float32."""
    chm_src = rasterio.open(os.path.join(RES, "detection", "chm.tif"))
    chm = chm_src.read(1).astype("float32")
    chm = np.nan_to_num(chm, nan=0.0).clip(0, 30) / 30.0
    with rasterio.open(os.path.join(DATA, "NAIP_image.tif")) as naip:
        with WarpedVRT(naip, crs=chm_src.crs, transform=chm_src.transform,
                       width=chm_src.width, height=chm_src.height,
                       resampling=Resampling.bilinear) as vrt:
            rgbn = vrt.read().astype("float32") / 255.0
    return np.concatenate([rgbn, chm[None]], 0), chm_src


class UNet(nn.Module):
    def __init__(self, ch=5, base=16):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(True), nn.Conv2d(o, o, 3, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(True))
        self.e1, self.e2, self.e3 = block(ch, base), block(base, base*2), block(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d2, self.d1 = block(base*6, base*2), block(base*3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        return self.head(d1)


def patch_window(src, cx, cy):
    row, col = src.index(cx, cy)
    r0 = int(np.clip(row - PATCH // 2, 0, src.height - PATCH))
    c0 = int(np.clip(col - PATCH // 2, 0, src.width - PATCH))
    return r0, c0


def main():
    stack, chm_src = load_stack()
    gaps = gpd.read_file(os.path.join(OUT, "gaps_labeled.geojson")).to_crs(CRS)
    o26 = gpd.read_file(os.path.join(DATA, "osm_buildings_2026.geojson")).to_crs(CRS)
    o26["geometry"] = o26.geometry.buffer(0)
    labels = rasterize(((g, 1) for g in o26.geometry if not g.is_empty),
                       out_shape=stack.shape[1:], transform=chm_src.transform,
                       fill=0, dtype="uint8")

    def patch(gap):
        c = gap.geometry.centroid
        r0, c0 = patch_window(chm_src, c.x, c.y)
        return (stack[:, r0:r0+PATCH, c0:c0+PATCH],
                labels[r0:r0+PATCH, c0:c0+PATCH], (r0, c0))

    train_rows = gaps[~gaps.east & gaps.filled]
    X, Y = [], []
    for _, gap in train_rows.iterrows():
        x, y, _ = patch(gap)
        for k in range(4):                      # rotations x flips = 8 augments
            xr, yr = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            X += [xr.copy(), np.flip(xr, 2).copy()]
            Y += [yr.copy(), np.flip(yr, 1).copy()]
    X = torch.tensor(np.stack(X))
    Y = torch.tensor(np.stack(Y)).float()
    print(f"device={DEV}  train patches={len(X)} (from {len(train_rows)} west gaps)")

    net = UNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    hist = []
    for ep in range(EPOCHS):
        net.train()
        perm, tot = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), 16):
            b = perm[i:i+16]
            opt.zero_grad()
            loss = lossf(net(X[b].to(DEV)).squeeze(1), Y[b].to(DEV))
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        hist.append(round(tot / len(X), 4))
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1}/{EPOCHS}  bce={hist[-1]}")

    # inference on every gap: pick the predicted component over the gap footprint
    net.eval()
    geoms, confs, ok_flags, iou_px = [], [], [], []
    with torch.no_grad():
        for _, gap in gaps.iterrows():
            x, y, (r0, c0) = patch(gap)
            prob = torch.sigmoid(net(torch.tensor(x[None]).to(DEV))).cpu().numpy()[0, 0]
            mask = (prob >= 0.5).astype("uint8")
            if gap.east and gap.filled:          # held-out pixel IoU
                inter, union = (mask & y).sum(), (mask | y).sum()
                iou_px.append(inter / union if union else 0.0)
            tr = rasterio.windows.transform(
                rasterio.windows.Window(c0, r0, PATCH, PATCH), chm_src.transform)
            cand = [shp_shape(s) for s, v in shapes(mask, transform=tr) if v == 1]
            cand = [p for p in cand if p.intersects(gap.geometry)]
            if not cand:                          # no prediction -> LiDAR fallback
                poly, ok = regularize(gap.geometry)
                geoms.append(poly), confs.append(0.0), ok_flags.append(False)
                continue
            comp = largest_part(max(cand, key=lambda p: p.area))
            m = ~rasterio.features.geometry_mask([comp], out_shape=mask.shape,
                                                 transform=tr)
            poly, ok = regularize(comp)
            geoms.append(poly)
            confs.append(round(float(prob[m].mean()), 3) if m.any() else 0.0)
            ok_flags.append(ok)

    proposals = gpd.GeoDataFrame(
        {"gap_id": gaps.gap_id, "building": "yes", "height": gaps.height_m.round(1),
         "source": "U-Net(NAIP+CHM) trained on OSM-2026 community polygons",
         "confidence": confs, "regularized": ok_flags,
         "filled": gaps.filled, "east": gaps.east},
        geometry=geoms, crs=CRS)
    proposals.to_crs(4326).to_file(os.path.join(OUT, "proposals_learned.geojson"),
                                   driver="GeoJSON")
    torch.save(net.state_dict(), os.path.join(OUT, "unet_weights.pt"))

    metrics = dict(device=DEV, epochs=EPOCHS, train_patches=int(len(X)),
                   train_gaps=int(len(train_rows)), bce_curve=hist,
                   east_pixel_iou_median=round(float(np.median(iou_px)), 3),
                   fallbacks=int(sum(1 for f, c in zip(ok_flags, confs)
                                     if not f and c == 0.0)))
    json.dump(metrics, open(os.path.join(OUT, "unet_metrics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "bce_curve"}, indent=2))


if __name__ == "__main__":
    main()

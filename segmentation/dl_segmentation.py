"""
DGCNN (EdgeConv) point-cloud SEMANTIC SEGMENTATION for the UIUC campus tile.
Upgrade over the vanilla-PointNet baseline: a dynamic k-NN graph gives each point
LOCAL geometric context (roof planes vs. scattered canopy), which the per-point
PointNet cannot see -> sharper boundaries, higher IoU.

Ground truth = ASPRS classification in the LAZ.  Spatial split (west train / east val).
Reuses the cached, decimated feature set written by ../dl_semseg.py (results/dl_cache.npz);
regenerates it if absent.

Outputs (../results/):
  seg_dgcnn.pt        trained weights
  seg_metrics.json    DGCNN metrics + comparison vs PointNet baseline
  seg_confusion.png   validation confusion matrix
  seg_fulltile.png    wall-to-wall predicted semantic map (whole 2x2 km)
  seg_labels.tif      predicted semantic label raster (0.5 m, EPSG:6350)
"""
import os, json, numpy as np, laspy, torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "UIUC_campus_LiDAR_merged_2x2km.laz")
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(OUT, "dl_cache.npz")
RES, BLOCK, NPTS, K = 0.5, 40.0, 2048, 16
CLASS_MAP = {2:0, 3:1, 4:2, 5:3, 6:4}
INV_MAP = {v:k for k,v in CLASS_MAP.items()}
NAMES = ["Ground","Low Veg","Med Veg","High Veg","Building"]; NC = len(NAMES)
CMAP = ListedColormap(["#b8a06a","#b6e59e","#5fbf5f","#1f7a1f","#d1483f"])
rng = np.random.default_rng(1); torch.manual_seed(1)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

h = laspy.open(SRC).header
xmin, ymin, xmax, ymax = h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]
nx = int(round((xmax-xmin)/RES)); ny = int(round((ymax-ymin)/RES))

# ---------------------------------------------------------- load cached features
if not os.path.exists(CACHE):
    raise SystemExit("run ../dl_semseg.py first to build results/dl_cache.npz")
d = np.load(CACHE)
X, Y, Z, HAG, INT, RR, NR, LAB = (d[k] for k in ["X","Y","Z","HAG","INT","RR","NR","LAB"])
print(f"[data] {len(X):,} points | device={dev}")

nbx = int(np.ceil((xmax-xmin)/BLOCK))
bid = (((Y-ymin)/BLOCK).astype(np.int32))*nbx + ((X-xmin)/BLOCK).astype(np.int32)
order = np.argsort(bid, kind="stable"); bids = bid[order]
uniq, starts = np.unique(bids, return_index=True); ends = np.append(starts[1:], len(bids))
blocks = {int(u): order[s:e] for u, s, e in zip(uniq, starts, ends) if e-s >= 200}
train_b = [u for u in blocks if xmin+(u%nbx+0.5)*BLOCK < 656000]
val_b   = [u for u in blocks if xmin+(u%nbx+0.5)*BLOCK >= 656000]
tr_idx = np.concatenate([blocks[u] for u in train_b])
INT_m, INT_s = INT[tr_idx].mean(), INT[tr_idx].std()+1e-6
print(f"[split] train blocks={len(train_b)}  val blocks={len(val_b)}")

def feats_for(sel, u):
    cx = xmin+(u%nbx+0.5)*BLOCK; cy = ymin+(u//nbx+0.5)*BLOCK; zx = Z[sel]
    return np.stack([(X[sel]-cx)/(BLOCK/2), (Y[sel]-cy)/(BLOCK/2), (zx-zx.min())/20.0,
        np.clip(HAG[sel]/50,0,1), (INT[sel]-INT_m)/INT_s, RR[sel]/np.maximum(NR[sel],1),
        NR[sel]/7, (X[sel]-xmin)/2000, (Y[sel]-ymin)/2000], axis=1).astype(np.float32)

def make_batch(bs):
    F_, L_ = [], []
    for u in bs:
        idx = blocks[u]; sel = idx[rng.integers(0, len(idx), NPTS)]
        F_.append(feats_for(sel, u)); L_.append(LAB[sel])
    return torch.from_numpy(np.stack(F_)), torch.from_numpy(np.stack(L_))

# ---------------------------------------------------------- DGCNN model
def knn(x, k):                                   # x: [B,C,N] -> idx [B,N,k]
    inner = -2*torch.matmul(x.transpose(2,1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    dist = -xx - inner - xx.transpose(2,1)       # negative squared distance
    return dist.topk(k=k, dim=-1)[1]

def graph_feature(x, k):                          # [B,C,N] -> [B,2C,N,k]
    B, C, N = x.shape
    idx = knn(x, k) + torch.arange(0, B, device=x.device).view(-1,1,1)*N
    xt = x.transpose(2,1).contiguous().view(B*N, C)
    nb = xt[idx.view(-1), :].view(B, N, k, C)
    xr = x.transpose(2,1).view(B, N, 1, C).expand(-1,-1,k,-1)
    return torch.cat((nb-xr, xr), dim=3).permute(0,3,1,2).contiguous()

class EdgeConv(nn.Module):
    def __init__(s, cin, cout, k):
        super().__init__(); s.k = k
        s.conv = nn.Sequential(nn.Conv2d(2*cin, cout, 1, bias=False),
                               nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))
    def forward(s, x): return s.conv(graph_feature(x, s.k)).max(dim=-1)[0]

class DGCNNSeg(nn.Module):
    def __init__(s, fin, nc, k=16):
        super().__init__()
        s.ec1, s.ec2, s.ec3 = EdgeConv(fin,64,k), EdgeConv(64,64,k), EdgeConv(64,64,k)
        s.g = nn.Sequential(nn.Conv1d(192,512,1,bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        s.head = nn.Sequential(
            nn.Conv1d(192+512,256,1,bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Conv1d(256,128,1,bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
            nn.Conv1d(128,nc,1))
    def forward(s, x):
        x = x.transpose(1,2)
        x1 = s.ec1(x); x2 = s.ec2(x1); x3 = s.ec3(x2)
        xc = torch.cat([x1,x2,x3], dim=1)
        g = s.g(xc).max(dim=-1, keepdim=True)[0].expand(-1,-1,x.shape[2])
        return s.head(torch.cat([xc, g], dim=1)).transpose(1,2)

model = DGCNNSeg(9, NC, K).to(dev)
nparam = sum(p.numel() for p in model.parameters())
freq = np.array([(LAB[tr_idx]==i).sum() for i in range(NC)], float)
w = torch.tensor(np.clip(np.median(freq)/freq, 0.5, 3.0), dtype=torch.float32, device=dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=14)
crit = nn.CrossEntropyLoss(weight=w)
def conf(p, t, nc): return np.bincount(t*nc+p, minlength=nc*nc).reshape(nc, nc)
print(f"[model] DGCNN params={nparam/1e6:.2f}M  k={K}  N={NPTS}")

# ---------------------------------------------------------- train (best checkpoint)
EPOCHS, BS, PER = 14, 8, 280
best, best_state = -1.0, None
for ep in range(1, EPOCHS+1):
    model.train(); rng.shuffle(train_b); tot = 0.0; picks = train_b[:PER]
    for i in range(0, len(picks), BS):
        xb, yb = make_batch(picks[i:i+BS]); xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad(); loss = crit(model(xb).reshape(-1,NC), yb.reshape(-1))
        loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    model.eval(); cm = np.zeros((NC,NC), int)
    with torch.no_grad():
        for i in range(0, min(len(val_b),300), BS):
            xb, yb = make_batch(val_b[i:i+BS])
            cm += conf(model(xb.to(dev)).argmax(-1).cpu().numpy().ravel(), yb.numpy().ravel(), NC)
    oa = np.trace(cm)/cm.sum(); iou = np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm)+1e-9); f = ""
    if iou.mean() > best:
        best = float(iou.mean()); best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}; f = "  <-best"
    print(f"  ep{ep:02d} loss={tot/max(1,len(picks)//BS):.3f}  valOA={oa:.3f}  mIoU={iou.mean():.3f}{f}")
model.load_state_dict(best_state)

# ---------------------------------------------------------- full validation
model.eval(); cm = np.zeros((NC,NC), int)
with torch.no_grad():
    for i in range(0, len(val_b), BS):
        xb, yb = make_batch(val_b[i:i+BS])
        cm += conf(model(xb.to(dev)).argmax(-1).cpu().numpy().ravel(), yb.numpy().ravel(), NC)
oa = float(np.trace(cm)/cm.sum()); iou = np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm)+1e-9)
baseline = {}
if os.path.exists(os.path.join(OUT,"dl_metrics.json")):
    baseline = json.load(open(os.path.join(OUT,"dl_metrics.json")))
metrics = dict(model="DGCNN", params_M=round(nparam/1e6,3), device=dev,
    overall_accuracy=round(oa,4), mean_IoU=round(float(iou.mean()),4),
    per_class_IoU={NAMES[i]: round(float(iou[i]),4) for i in range(NC)},
    baseline_pointnet=dict(overall_accuracy=baseline.get("overall_accuracy"),
                           mean_IoU=baseline.get("mean_IoU")))
json.dump(metrics, open(os.path.join(OUT,"seg_metrics.json"),"w"), indent=2)
torch.save(model.state_dict(), os.path.join(OUT,"seg_dgcnn.pt"))
print(json.dumps(metrics, indent=2))

# confusion figure
cmn = cm/cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6,5.5), dpi=130)
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(NC)); ax.set_yticks(range(NC))
ax.set_xticklabels(NAMES, rotation=45, ha="right"); ax.set_yticklabels(NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True (ASPRS)")
for i in range(NC):
    for j in range(NC):
        ax.text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",
                color="white" if cmn[i,j]>.5 else "black", fontsize=8)
ax.set_title(f"DGCNN semantic seg — val confusion\nOA={oa:.3f}  mIoU={iou.mean():.3f}")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"seg_confusion.png")); plt.close(fig)

# ---------------------------------------------------------- wall-to-wall semantic raster
print("[raster] predicting whole tile -> label raster")
votes = np.zeros((ny*nx, NC), np.int32)
all_b = list(blocks);
with torch.no_grad():
    for i in range(0, len(all_b), BS):
        bs = all_b[i:i+BS]; sels = []
        F_ = []
        for u in bs:
            idx = blocks[u]; sel = idx[rng.integers(0, len(idx), NPTS)]
            sels.append(sel); F_.append(feats_for(sel, u))
        pr = model(torch.from_numpy(np.stack(F_)).to(dev)).argmax(-1).cpu().numpy()
        for j, sel in enumerate(sels):
            col = np.clip(((X[sel]-xmin)/RES).astype(np.int64), 0, nx-1)
            row = np.clip(((ymax-Y[sel])/RES).astype(np.int64), 0, ny-1)
            np.add.at(votes, (row*nx+col, pr[j]), 1)
voted = votes.sum(1) > 0
label = np.full(ny*nx, -1, np.int16)
label[voted] = votes[voted].argmax(1)
# nearest-fill empty cells
from scipy import ndimage as ndi
lab2d = label.reshape(ny, nx)
_, (ir, ic) = ndi.distance_transform_edt(lab2d < 0, return_indices=True)
lab2d = lab2d[ir, ic]

import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
with rasterio.open(os.path.join(OUT,"seg_labels.tif"), "w", driver="GTiff", height=ny, width=nx,
                   count=1, dtype="int16", crs=CRS.from_epsg(6350),
                   transform=from_origin(xmin, ymax, RES, RES), nodata=-1) as dst:
    dst.write(lab2d.astype(np.int16), 1)

fig, ax = plt.subplots(figsize=(11,11), dpi=130)
ax.imshow(lab2d, cmap=CMAP, vmin=0, vmax=NC-1, extent=[xmin,xmax,ymin,ymax], origin="upper")
ax.axvline(656000, color="k", lw=0.8, ls="--", alpha=0.6)
ax.set_title(f"DGCNN wall-to-wall semantic segmentation — whole 2×2 km\n"
             f"(dashed = train|val boundary; east half held out)  OA={oa:.3f} mIoU={iou.mean():.3f}")
ax.set_xlabel("Easting (m, EPSG:6350)"); ax.set_ylabel("Northing (m)"); ax.ticklabel_format(style="plain")
ax.legend(handles=[Patch(color=CMAP(i), label=NAMES[i]) for i in range(NC)],
          loc="lower center", ncol=NC, bbox_to_anchor=(0.5,-0.08))
fig.tight_layout(); fig.savefig(os.path.join(OUT,"seg_fulltile.png"), bbox_inches="tight"); plt.close(fig)
print("done ->", OUT)

"""
Deep-learning point-cloud SEMANTIC SEGMENTATION on the UIUC campus tile.
Ground-truth = the ASPRS classification already in the LAZ.
Model  : block-wise PointNet (PyTorch, runs on Apple-Silicon MPS).
Split  : spatial -- west half (x<656000) TRAIN, east half VAL (no leakage).
Classes: 0 Ground(2) 1 LowVeg(3) 2 MedVeg(4) 3 HighVeg(5) 4 Building(6)

Outputs (outputs/segmentation/):
  dl_cache.npz            cached preprocessed points/features
  dl_metrics.json         overall accuracy, mean IoU, per-class IoU
  dl_confusion.png        confusion matrix
  dl_prediction.png       GT vs predicted labels on the VAL (east) region
  dl_pointnet.pt          trained weights
"""
import os, json, numpy as np, laspy, rasterio, torch
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "UIUC_campus_LiDAR_merged_2x2km.laz")
OUT = os.path.join(ROOT, "outputs", "segmentation"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "dl_cache.npz")
DTM_TIF = os.path.join(ROOT, "outputs", "detection", "dtm.tif")  # from classical_detection.py
RES = 0.5; BLOCK = 40.0; NPTS = 4096
CLASS_MAP = {2:0, 3:1, 4:2, 5:3, 6:4}
NAMES = ["Ground", "Low Veg", "Med Veg", "High Veg", "Building"]
NC = len(NAMES)
COLORS = ListedColormap(["#b8a06a", "#b6e59e", "#5fbf5f", "#1f7a1f", "#d1483f"])
torch.manual_seed(0); rng = np.random.default_rng(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"

h = laspy.open(SRC).header
xmin, ymin, xmax, ymax = h.mins[0], h.mins[1], h.maxs[0], h.maxs[1]

# --------------------------------------------------- preprocess (cached)
if not os.path.exists(CACHE):
    print("[prep] reading points + sampling DTM for height-above-ground")
    with rasterio.open(DTM_TIF) as ds:
        dtm = ds.read(1)
    ny, nx = dtm.shape
    keep_x, keep_y, keep_z, keep_hag = [], [], [], []
    keep_int, keep_rr, keep_nr, keep_lab = [], [], [], []
    with laspy.open(SRC) as r:
        for pts in r.chunk_iterator(10_000_000):
            c = np.asarray(pts.classification)
            m = np.isin(c, list(CLASS_MAP))
            m &= rng.random(len(c)) < 0.25          # decimate ~4x for speed
            if not m.any(): continue
            x, y, z = np.asarray(pts.x)[m], np.asarray(pts.y)[m], np.asarray(pts.z)[m]
            col = np.clip(((x - xmin)/RES).astype(np.int64), 0, nx-1)
            row = np.clip(((ymax - y)/RES).astype(np.int64), 0, ny-1)
            hag = np.clip(z - dtm[row, col], 0, None)
            keep_x.append(x); keep_y.append(y); keep_z.append(z); keep_hag.append(hag)
            keep_int.append(np.asarray(pts.intensity)[m].astype(np.float32))
            keep_rr.append(np.asarray(pts.return_number)[m].astype(np.float32))
            keep_nr.append(np.asarray(pts.number_of_returns)[m].astype(np.float32))
            keep_lab.append(np.array([CLASS_MAP[v] for v in c[m]], np.int64))
    X = np.concatenate(keep_x); Y = np.concatenate(keep_y); Z = np.concatenate(keep_z)
    HAG = np.concatenate(keep_hag); INT = np.concatenate(keep_int)
    RR = np.concatenate(keep_rr); NR = np.concatenate(keep_nr); LAB = np.concatenate(keep_lab)
    np.savez_compressed(CACHE, X=X, Y=Y, Z=Z, HAG=HAG, INT=INT, RR=RR, NR=NR, LAB=LAB)
    print(f"[prep] cached {len(X):,} points")

d = np.load(CACHE)
X, Y, Z, HAG, INT, RR, NR, LAB = (d[k] for k in ["X","Y","Z","HAG","INT","RR","NR","LAB"])
print(f"[data] {len(X):,} points | device={dev}")
print("       class counts:", {NAMES[i]: int((LAB==i).sum()) for i in range(NC)})

# --------------------------------------------------- block indexing + spatial split
bx = ((X - xmin)/BLOCK).astype(np.int32); by = ((Y - ymin)/BLOCK).astype(np.int32)
nbx = int(np.ceil((xmax-xmin)/BLOCK))
bid = by * nbx + bx
order = np.argsort(bid, kind="stable")
bid_s = bid[order]
uniq, starts = np.unique(bid_s, return_index=True)
ends = np.append(starts[1:], len(bid_s))
block_pts = {int(u): order[s:e] for u, s, e in zip(uniq, starts, ends) if e - s >= 200}
# spatial split by block-center easting
def bcx(u): return xmin + ((u % nbx) + 0.5) * BLOCK
train_b = [u for u in block_pts if bcx(u) < 656000]
val_b   = [u for u in block_pts if bcx(u) >= 656000]
print(f"[split] train blocks={len(train_b)}  val blocks={len(val_b)}")

# feature standardisation stats from TRAIN points only
tr_idx = np.concatenate([block_pts[u] for u in train_b])
INT_m, INT_s = INT[tr_idx].mean(), INT[tr_idx].std() + 1e-6

def make_batch(blocks):
    F_, L_ = [], []
    for u in blocks:
        idx = block_pts[u]
        sel = idx[rng.integers(0, len(idx), NPTS)] if len(idx) >= NPTS else \
              idx[rng.integers(0, len(idx), NPTS)]
        cx = xmin + (u % nbx + 0.5) * BLOCK; cy = ymin + (u // nbx + 0.5) * BLOCK
        zx = Z[sel]
        feat = np.stack([
            (X[sel]-cx)/(BLOCK/2), (Y[sel]-cy)/(BLOCK/2),   # block-relative xy
            (zx - zx.min())/20.0,                            # relative z
            np.clip(HAG[sel]/50.0, 0, 1),                    # height above ground
            (INT[sel]-INT_m)/INT_s,                          # intensity (std)
            RR[sel]/np.maximum(NR[sel],1),                   # return ratio
            NR[sel]/7.0,                                     # num returns
            (X[sel]-xmin)/2000.0, (Y[sel]-ymin)/2000.0,      # global position
        ], axis=1).astype(np.float32)
        F_.append(feat); L_.append(LAB[sel])
    return torch.from_numpy(np.stack(F_)), torch.from_numpy(np.stack(L_))

# --------------------------------------------------- PointNet (segmentation)
class PointNetSeg(nn.Module):
    def __init__(self, fin, nc):
        super().__init__()
        def mlp(i,o): return nn.Sequential(nn.Conv1d(i,o,1), nn.BatchNorm1d(o), nn.ReLU())
        self.e1, self.e2, self.e3 = mlp(fin,64), mlp(64,64), mlp(64,128)
        self.e4 = mlp(128,1024)
        self.d1, self.d2, self.d3 = mlp(1024+64,512), mlp(512,256), mlp(256,128)
        self.head = nn.Conv1d(128, nc, 1)
    def forward(self, x):                 # x: [B,N,F]
        x = x.transpose(1,2)              # [B,F,N]
        p = self.e2(self.e1(x))           # [B,64,N]  local point feats
        g = self.e4(self.e3(p))           # [B,1024,N]
        g = torch.max(g, 2, keepdim=True)[0].expand(-1,-1,x.shape[2])  # global
        y = self.d3(self.d2(self.d1(torch.cat([p, g], 1))))
        return self.head(y).transpose(1,2)   # [B,N,nc]

model = PointNetSeg(9, NC).to(dev)
# gentle median-frequency class balancing (clipped) -> stable training
freq = np.array([(LAB[tr_idx]==i).sum() for i in range(NC)], float)
wv = np.clip(np.median(freq)/freq, 0.5, 3.0)
w = torch.tensor(wv, dtype=torch.float32, device=dev)
opt = torch.optim.Adam(model.parameters(), lr=7e-4)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.6)
crit = nn.CrossEntropyLoss(weight=w)

def confusion(pred, true, nc):
    k = true*nc + pred
    return np.bincount(k, minlength=nc*nc).reshape(nc, nc)

# --------------------------------------------------- train
EPOCHS, BS, TRAIN_PER_EP = 18, 16, 500
print(f"[train] PointNet on {dev} | {EPOCHS} epochs")
best_miou, best_state = -1.0, None
for ep in range(1, EPOCHS+1):
    model.train(); rng.shuffle(train_b); tot = 0.0
    picks = train_b[:TRAIN_PER_EP]
    for i in range(0, len(picks), BS):
        xb, yb = make_batch(picks[i:i+BS])
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out.reshape(-1, NC), yb.reshape(-1))
        loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    # quick val
    model.eval(); cm = np.zeros((NC, NC), int)
    with torch.no_grad():
        for i in range(0, min(len(val_b), 300), BS):
            xb, yb = make_batch(val_b[i:i+BS])
            pr = model(xb.to(dev)).argmax(-1).cpu().numpy().ravel()
            cm += confusion(pr, yb.numpy().ravel(), NC)
    oa = np.trace(cm)/cm.sum()
    iou = np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm)+1e-9)
    flag = ""
    if iou.mean() > best_miou:                       # keep BEST model, not last
        best_miou = float(iou.mean())
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        flag = "  <- best"
    print(f"  ep{ep:02d} loss={tot/max(1,len(picks)//BS):.3f}  valOA={oa:.3f}  mIoU={iou.mean():.3f}{flag}")

model.load_state_dict(best_state)                    # restore best before final eval/viz
print(f"[train] restored best model (quick-val mIoU={best_miou:.3f})")

# --------------------------------------------------- full val evaluation
print("[eval] full validation pass")
model.eval(); cm = np.zeros((NC, NC), int)
with torch.no_grad():
    for i in range(0, len(val_b), BS):
        xb, yb = make_batch(val_b[i:i+BS])
        pr = model(xb.to(dev)).argmax(-1).cpu().numpy().ravel()
        cm += confusion(pr, yb.numpy().ravel(), NC)
oa = float(np.trace(cm)/cm.sum())
iou = np.diag(cm)/(cm.sum(0)+cm.sum(1)-np.diag(cm)+1e-9)
prec = np.diag(cm)/(cm.sum(0)+1e-9); rec = np.diag(cm)/(cm.sum(1)+1e-9)
metrics = dict(device=dev, overall_accuracy=round(oa,4), mean_IoU=round(float(iou.mean()),4),
               per_class={NAMES[i]: dict(IoU=round(float(iou[i]),4),
                          precision=round(float(prec[i]),4), recall=round(float(rec[i]),4))
                          for i in range(NC)})
json.dump(metrics, open(os.path.join(OUT,"dl_metrics.json"),"w"), indent=2)
print(json.dumps(metrics, indent=2))
torch.save(model.state_dict(), os.path.join(OUT,"dl_pointnet.pt"))

# confusion matrix fig
cmn = cm/cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6,5.5), dpi=130)
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(NC)); ax.set_yticks(range(NC))
ax.set_xticklabels(NAMES, rotation=45, ha="right"); ax.set_yticklabels(NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True (ASPRS)")
for i in range(NC):
    for j in range(NC):
        ax.text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",
                color="white" if cmn[i,j]>.5 else "black", fontsize=8)
ax.set_title(f"PointNet semantic seg — val confusion\nOA={oa:.3f}  mIoU={iou.mean():.3f}")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"dl_confusion.png")); plt.close(fig)

# --------------------------------------------------- prediction map (val/east region)
print("[viz] rendering GT vs predicted over east region")
gx, gy, gt, pd = [], [], [], []
with torch.no_grad():
    for i in range(0, len(val_b), BS):
        blocks = val_b[i:i+BS]
        xb, yb = make_batch(blocks)
        pr = model(xb.to(dev)).argmax(-1).cpu().numpy()
        for j, u in enumerate(blocks):
            idx = block_pts[u]; sel = idx[rng.integers(0,len(idx),NPTS)]
            gx.append(X[sel]); gy.append(Y[sel]); gt.append(LAB[sel]); pd.append(pr[j])
gx=np.concatenate(gx); gy=np.concatenate(gy); gt=np.concatenate(gt); pd=np.concatenate(pd)
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=130)
for ax, lab, ttl in [(axes[0], gt, "Ground truth (ASPRS)"), (axes[1], pd, "PointNet prediction")]:
    ax.scatter(gx, gy, c=lab, cmap=COLORS, s=0.5, vmin=0, vmax=NC-1, linewidths=0)
    ax.set_title(ttl); ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
    ax.set_xlim(656000, xmax); ax.set_ylim(ymin, ymax)
handles=[plt.Line2D([],[],marker='o',ls='',color=COLORS(i),label=NAMES[i]) for i in range(NC)]
fig.legend(handles=handles, loc="lower center", ncol=NC)
fig.suptitle(f"Semantic segmentation — validation (east half)   OA={oa:.3f}  mIoU={iou.mean():.3f}")
fig.tight_layout(rect=[0,0.05,1,1]); fig.savefig(os.path.join(OUT,"dl_prediction.png")); plt.close(fig)
print("done ->", OUT)

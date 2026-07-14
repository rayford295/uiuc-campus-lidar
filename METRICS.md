# Comparison Metrics — OSM vs LiDAR vs NAIP

Metrics for comparing the three sources, organized by **level**, from "is this one building
right?" up to "is the under-mapping spatially systematic?" (the latter is the core research
contribution). Each metric notes what it answers, whether it serves **detection** or
**correction**, and its implementation status.

## Ground truth definition
The three sources are **not** treated symmetrically. Ground truth is the **RS consensus**:
a building is "real" when **LiDAR ∩ NAIP** both indicate it (two independent sensors agree).
OSM is then evaluated against that consensus. This is what makes NAIP's role (corroborating
LiDAR) meaningful rather than a redundant third comparison.

---

## Level 1 · Object-level (per building) — detect omissions / commissions
| Metric | Meaning | Status |
|--------|---------|--------|
| **Completeness / Recall** = matched / total RS-truth | how many real buildings OSM is missing (core) | ✅ |
| **Correctness / Precision** = RS-supported OSM / all OSM | how many OSM buildings are spurious / stale | ✅ |
| **F1** | harmonic mean of the two | easy add |
| **Count ratio** N_OSM / N_RS | under-mapping at the count level | ✅ |
| **1:N match ratio** | OSM merging/splitting buildings (generalization differences) | easy add |

Matching is IoU-based (threshold 0.3); one-to-one and one-to-many cases are tracked.

## Level 2 · Geometric accuracy (matched objects) — VGI positional / shape quality
- **IoU distribution** (median / mean) of matched pairs. ✅
- **Area agreement**: total-area ratio, per-building area error, and systematic bias (does OSM consistently draw footprints too large / small?).
- **Positional accuracy**: centroid offset magnitude *and direction* (detects systematic shift).
- **Boundary displacement / Hausdorff distance**: outline fidelity.
- **Shape similarity**: perimeter ratio, compactness.

## Level 3 · Pixel-level — implemented
Overall accuracy, **building-pixel IoU**, **Cohen's κ**, TP/FP/FN areas, edge-vs-interior
disagreement. ✅ (`src/pixel_comparison.py`, 0.1/0.2 m)

## Level 4 · Spatial bias ⭐ (the heart of "detecting bias")
This is where the study moves from "OSM is X% complete" to "**completeness varies
systematically across space and correlates with who lives there**":
- **Gridded completeness surface** (per-cell recall) = the bias map. ✅ (framework)
- **Moran's I** — is under-mapping *spatially clustered*? (bias is inherently spatial)
- **Getis-Ord Gi\*** — hot/cold spots of omission.
- **Completeness ~ covariates regression** — regress per-cell completeness on population
  density, income, race, urban/rural, distance-to-core, road density → coefficients + R²
  reveal what drives the bias. *(needs Census / ACS)*
- **Equity indices** — completeness gap across income/demographic groups; **Gini** inequality.

## Level 5 · Three-source consensus / confidence
Exploits having **two independent RS references**:
- **Three-way voting matrix** — per object/pixel, which subset of {OSM, LiDAR, NAIP} calls
  it building: all-three = certain; **RS-both & not-OSM = high-confidence OSM omission**;
  OSM-only = suspected OSM error / stale.
- **Pairwise κ** (OSM-LiDAR, OSM-NAIP, LiDAR-NAIP).
- **LiDAR–NAIP agreement as a confidence weight** — trust OSM judgements more where the two
  sensors agree.

## Level 6 · VGI-specific (attributes + currency)
- **Attribute completeness** — do OSM buildings carry `building=type`, `height`, `name`?
  (VGI gaps are not only geometric.)
- **Height accuracy** — where OSM tags `height`, compare against LiDAR per-building height.
- **Mapping lag** — features in RS-2019 that appear in OSM only by 2026: quantify how long
  omissions take to be filled (uses both OSM snapshots; ties to "buildings/roads change slowly").

## Roads (OSM vs NAIP, linear features)
- **Network length completeness** (matched length / total), **buffer overlap ratio**,
  **topology** (intersection count, connectivity).

## Correction effectiveness (does fixing the bias help?)
After augmenting OSM with RS-consensus features: completeness gain, and **reduction in
spatial inequality** (Gini / Moran's I) — evidence that the correction narrows the bias.

---

## Recommended minimum publishable set
1. **Recall + Precision + F1** (object level)
2. **Gridded completeness surface + Moran's I** (bias is spatial)
3. **Completeness ~ socioeconomic covariates regression** (bias is systematic & has social
   meaning — this is what elevates "incomplete map" to "digital divide / equity")
4. **Three-way voting** (demonstrates the value of the three-source design)

## Status
- Levels 1–3: code essentially ready (`src/vgi_comparison.py`, `src/pixel_comparison.py`) — run on real OSM to produce numbers.
- Level 4: needs (a) Census/ACS covariates and (b) scaling the study area to an
  **urban→rural gradient** (the campus tile is too uniform — completeness ≈ 100%, no Moran's I signal).
- Levels 5–6: straightforward to add.

See [METHODOLOGY.md](METHODOLOGY.md) for the overall design.

"""Statewide VGI spatial-bias analysis — OSM 2019 IL major roads.

Input: gis_osm_roads_2019_IL_Major_Roads.shp (375,754 segments, county-joined)
Auxiliary: Census 2019 gazetteer (county land area), co-est2019 (population),
cb_2019_us_county_20m (boundaries for mapping).

County-level bias metrics:
  - attribute completeness: % segments tagged with maxspeed / surface / ref / name
  - edit recency: median lastchange year, % edited 2017+
  - supply: road length density (km/km^2), km per 1,000 residents
Bias tests: Spearman correlation of each metric vs. population density,
plus urban/rural contrast (>=100 vs <100 persons/km^2).

Usage: python src/statewide_bias.py <roads_shp> <aux_dir> <out_dir>
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
C_URBAN, C_RURAL = "#2a78d6", "#eb6834"  # categorical slots 1 and 6
TEXT, MUTED = "#0b0b0b", "#52514e"

URBAN_THRESHOLD = 100  # persons/km^2


def nonempty(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")


def load_county_aux(aux: Path) -> pd.DataFrame:
    gaz = pd.read_csv(aux / "gaz_counties_17.txt", sep="\t")
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz["GEOID"] = gaz["GEOID"].astype(str)
    gaz["aland_km2"] = gaz["ALAND"] / 1e6

    pop = pd.read_csv(aux / "co-est2019.csv", encoding="latin-1")
    pop = pop[(pop.STATE == 17) & (pop.COUNTY > 0)].copy()
    pop["GEOID"] = pop.STATE.astype(str) + pop.COUNTY.astype(str).str.zfill(3)
    pop = pop[["GEOID", "CTYNAME", "POPESTIMATE2019"]]

    df = gaz[["GEOID", "NAME", "aland_km2"]].merge(pop, on="GEOID")
    df["pop_density"] = df.POPESTIMATE2019 / df.aland_km2
    return df


def county_metrics(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = gdf.copy()
    gdf["GEOID"] = "17" + gdf.CO_FIPS.astype(int).astype(str).str.zfill(3)
    gdf["len_km"] = gdf.geometry.to_crs(5070).length / 1000
    gdf["speed_ok"] = pd.to_numeric(gdf.maxspeed, errors="coerce").fillna(0) > 0
    gdf["surface_ok"] = nonempty(gdf.surface)
    gdf["ref_ok"] = nonempty(gdf.ref)
    gdf["name_ok"] = nonempty(gdf.loc_name)
    gdf["edit_year"] = pd.to_datetime(gdf.lastchange, errors="coerce").dt.year
    gdf["recent_edit"] = gdf.edit_year >= 2017

    agg = gdf.groupby("GEOID").agg(
        n_segments=("osm_id", "size"),
        total_km=("len_km", "sum"),
        pct_maxspeed=("speed_ok", "mean"),
        pct_surface=("surface_ok", "mean"),
        pct_ref=("ref_ok", "mean"),
        pct_name=("name_ok", "mean"),
        median_edit_year=("edit_year", "median"),
        pct_recent=("recent_edit", "mean"),
    )
    for c in ["pct_maxspeed", "pct_surface", "pct_ref", "pct_name", "pct_recent"]:
        agg[c] *= 100
    return agg.reset_index()


def main(roads_shp: str, aux_dir: str, out_dir: str) -> None:
    aux, out = Path(aux_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = ["osm_id", "fclass", "lastchange", "maxspeed", "surface", "ref",
            "loc_name", "CO_FIPS"]
    try:
        roads = gpd.read_file(roads_shp, columns=cols)
    except TypeError:  # old geopandas / fiona engine has no `columns=`
        roads = gpd.read_file(roads_shp)[cols + ["geometry"]]
    metrics = county_metrics(roads)
    counties = load_county_aux(aux)
    df = metrics.merge(counties, on="GEOID")
    df["road_density"] = df.total_km / df.aland_km2
    df["km_per_1k"] = df.total_km / (df.POPESTIMATE2019 / 1000)
    df["urban"] = df.pop_density >= URBAN_THRESHOLD
    df.to_csv(out / "county_metrics.csv", index=False)

    # ---- bias tests -------------------------------------------------------
    tests = []
    for m in ["pct_maxspeed", "pct_surface", "pct_ref", "pct_name",
              "pct_recent", "median_edit_year", "road_density", "km_per_1k"]:
        rho, p = spearmanr(df.pop_density, df[m])
        tests.append({"metric": m, "spearman_rho": rho, "p_value": p,
                      "urban_mean": df.loc[df.urban, m].mean(),
                      "rural_mean": df.loc[~df.urban, m].mean()})
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "bias_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))

    # ---- figures ----------------------------------------------------------
    shp = gpd.read_file(aux / "cb_county" / "cb_2019_us_county_20m.shp")
    il = shp[shp.STATEFP == "17"].merge(df, on="GEOID").to_crs(26971)

    panels = [
        ("pct_maxspeed", "Maxspeed tagged (%)"),
        ("pct_surface", "Surface tagged (%)"),
        ("median_edit_year", "Median last-edit year"),
        ("pct_recent", "Edited 2017+ (%)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    for ax, (col, title) in zip(axes, panels):
        # cap at the 95th percentile so a single outlier county can't flatten the ramp
        vmax = il[col].quantile(0.95) if col.startswith("pct_") else il[col].max()
        il.plot(column=col, cmap=CMAP, linewidth=0.3, edgecolor="#fcfcfb",
                legend=True, ax=ax, vmax=vmax, legend_kwds={"shrink": 0.6, "extend": "max"})
        ax.set_title(title, fontsize=11, color=TEXT)
        ax.set_axis_off()
    fig.suptitle("OSM 2019 Illinois roads — county-level completeness & recency",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(out / "choropleth_completeness.png", dpi=150,
                facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    scatters = [("pct_maxspeed", "Maxspeed tagged (%)"),
                ("pct_recent", "Edited 2017+ (%)"),
                ("median_edit_year", "Median last-edit year")]
    for ax, (m, label) in zip(axes, scatters):
        for flag, c, lab in [(True, C_URBAN, f"urban (>{URBAN_THRESHOLD}/km$^2$)"),
                             (False, C_RURAL, "rural")]:
            sub = df[df.urban == flag]
            ax.scatter(sub.pop_density, sub[m], s=34, c=c, label=lab,
                       edgecolors="#fcfcfb", linewidths=0.8, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Population density (persons/km$^2$, log)", color=MUTED)
        ax.set_ylabel(label, color=MUTED)
        rho, p = spearmanr(df.pop_density, df[m])
        ax.set_title(f"$\\rho$ = {rho:.2f}  (p = {p:.1e})", fontsize=11, color=TEXT)
        ax.grid(alpha=0.25, linewidth=0.5)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Attribute completeness & edit recency vs. population density",
                 fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(out / "scatter_bias.png", dpi=150, facecolor="#fcfcfb",
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out}/county_metrics.csv, bias_tests.csv, 2 figures")


if __name__ == "__main__":
    main(*sys.argv[1:4])

"""Statewide deployment priority — where would automated correction pay off?

Combines the statewide bias metrics (results/statewide/county_metrics.csv)
into a single per-county priority score:

    priority = staleness x population exposure
    staleness  = 1 - (% segments edited 2017+)/100      (how unmaintained)
    exposure   = min-max normalized log10(population)   (how many people affected)

Both factors are in [0, 1]; a county ranks high only when its OSM is stale AND
many residents depend on it — exactly where a remote-sensing-driven correction
pipeline (or a targeted mapping campaign) should be pointed first.

Usage:  python src/deploy_priority.py
Writes results/correction/deploy_priority.{png,csv} (+ top-10 in the JSON).
"""
import json
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "correction")
os.makedirs(OUT, exist_ok=True)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def main():
    df = pd.read_csv(os.path.join(ROOT, "results", "statewide",
                                  "county_metrics.csv"),
                     dtype={"GEOID": str})
    df["staleness"] = 1 - df.pct_recent / 100
    logpop = np.log10(df.POPESTIMATE2019)
    df["exposure"] = (logpop - logpop.min()) / (logpop.max() - logpop.min())
    df["priority"] = (df.staleness * df.exposure).round(4)
    df = df.sort_values("priority", ascending=False)

    cols = ["GEOID", "CTYNAME", "POPESTIMATE2019", "pct_recent",
            "median_edit_year", "staleness", "exposure", "priority"]
    df[cols].to_csv(os.path.join(OUT, "deploy_priority.csv"), index=False)
    top10 = df[cols].head(10).to_dict("records")
    json.dump(dict(top10=top10), open(os.path.join(OUT,
              "deploy_priority_top10.json"), "w"), indent=2, default=str)

    shp = gpd.read_file(os.path.join(ROOT, "data", "statewide", "cb_county",
                                     "cb_2019_us_county_20m.shp"))
    il = shp[shp.STATEFP == "17"].merge(df, on="GEOID").to_crs(26971)
    fig, ax = plt.subplots(figsize=(8.5, 10))
    il.plot(column="priority", cmap=LinearSegmentedColormap.from_list("b", SEQ),
            linewidth=0.3, edgecolor="#fcfcfb", legend=True, ax=ax,
            legend_kwds={"shrink": 0.6, "label": "correction priority"})
    for _, r in il.nlargest(6, "priority").iterrows():
        c = r.geometry.centroid
        ax.annotate(r.CTYNAME.replace(" County", ""), (c.x, c.y),
                    ha="center", fontsize=8, color="#0b0b0b")
    ax.set_title("Where automated VGI correction pays off first\n"
                 "(staleness × population exposure, OSM 2019 Illinois)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "deploy_priority.png"), dpi=150,
                facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)

    print(pd.DataFrame(top10)[["CTYNAME", "POPESTIMATE2019", "pct_recent",
                               "priority"]].to_string(index=False))
    print("->", os.path.join(OUT, "deploy_priority.png"))


if __name__ == "__main__":
    main()

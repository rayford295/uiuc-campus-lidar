# Project Description — Evaluating & Calibrating VGI Maps with Multimodal Remote Sensing

Volunteered Geographic Information (VGI) maps (e.g., OpenStreetMap, Mapillary) are
widely used for urban analytics, disaster response, and environmental applications.
However, its quality is uneven: while highly accurate in developed urban areas, VGI often
suffers from incompleteness and positional errors in rural regions and the Global South
due to limited contributions and expert effort in calibration. This spatial bias can
introduce uncertainty into downstream analysis, particularly in data-sparse regions.

This project aims to develop a systematic approach to evaluate and calibrate VGI maps
using multimodal remote sensing data (e.g., Landsat satellite remote sensing imagery and
LiDAR remote sensing data). The key research questions are:

1. How does VGI accuracy and completeness vary across geographic and socioeconomic contexts?
2. Can remote sensing data detect discrepancies in VGI data such as roads and buildings?
3. How can we develop scalable (AI) approaches to automatically improve VGI quality in data-sparse regions?

During the Summer School, the team will collaborate to design evaluation metrics and
workflows, extract features from imagery and LiDAR, and develop models for detection and
calibration. We will design a scalable workflow for benchmarking and enhancing VGI data
quality across diverse geographic contexts.

## Software applications and libraries (with versions)

**GIS software**

- ArcGIS Pro
- ENVI
- QGIS

**Python libraries**

- Geospatial: `geopandas 1.1.3`, `shapely 2.1.2`, `rasterio 1.5.0`, `pyproj 3.7.2`
- VGI: `osmnx 2.1.0`
- LiDAR: `laspy 2.5.0`
- Machine learning: `scikit-learn 1.8.0`, `pytorch 2.x`, `tensorflow 2.x`
- Base scientific stack: `matplotlib`, `numpy`, `scipy`, etc.

## Languages (with versions)

- Python 3.x
- JSON
- Markdown

## Data sets

- LiDAR data — [USGS 3D Elevation Program (3DEP)](https://www.usgs.gov/3d-elevation-program)
- Landsat remote sensing — [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- VGI maps — [OpenStreetMap](https://www.openstreetmap.org)
  - 2019 Illinois statewide extracts (buildings + roads) are archived under this repo's
    [`osm-il-2019` release](https://github.com/rayford295/uiuc-campus-lidar/releases/tag/osm-il-2019);
    campus-extent subsets are committed at the repo root
    (`osm_buildings_2019.geojson`, `osm_roads_2019.geojson`)
- Census data — [American Community Survey (ACS)](https://www.census.gov/programs-surveys/acs.html)

## Good fit for students interested in

- Geospatial applications of machine learning
- Volunteered Geographic Information (VGI) maps
- Remote sensing
- Data fusion

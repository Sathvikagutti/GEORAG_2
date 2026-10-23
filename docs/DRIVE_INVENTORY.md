# GeoRAG — Verified Google Drive Inventory

Folder inspected: https://drive.google.com/drive/folders/1sYhSXhyMs3OVyt1wGEUoidcpEd2gNUci
Inspected on 2026-10-22 by listing the folder pages and recursively listing every
subfolder. **All files below were actually seen and (where downloaded) opened — nothing
in this table is assumed.** Local copies live under `drive_data/`, `drive_downloads/`
and `extracted/`; the Drive itself was never modified.

## 1. Root folder (12 items — complete listing)

| File or folder | Exact path (Drive) | File type | Purpose | Verified status |
|---|---|---|---|---|
| GeoRAG_Indices | `/GeoRAG_Indices` | folder | spectral index rasters per year | Listed, opened ✓ |
| GeoRAG_LandUse_Labels | `/GeoRAG_LandUse_Labels` | folder | land-use label rasters per year | Listed, opened ✓ |
| GeoRAG_Satellite | `/GeoRAG_Satellite` | folder | satellite scenes per year + boundary | Listed, opened ✓ |
| Historical flood map | `/Historical flood map` | PDF, 4.7 MB | flood mapping (other member's topic) | Listed only (not needed for Member 2) |
| india-150101.osm Water.pbf | `/india-150101.osm Water.pbf` | OSM extract, 157.7 MB | OpenStreetMap water data 2015 | Listed only (unrelated to Member 2) |
| india-150101.osm.pbf | `/india-150101.osm.pbf` | OSM extract, 157.7 MB | OSM data 2015 | Listed only |
| india-200101.osm Water.pbf | `/india-200101.osm Water.pbf` | OSM extract, 649.1 MB | OSM water 2020 | Listed only |
| india-200101.osm.pbf | `/india-200101.osm.pbf` | OSM extract, 649.1 MB | OSM data 2020 | Listed only |
| india-250101.osm Water.pbf | `/india-250101.osm Water.pbf` | OSM extract, 1.46 GB | OSM water 2025 | Listed only |
| india-250101.osm.pbf | `/india-250101.osm.pbf` | OSM extract, 1.46 GB | OSM data 2025 | Listed only |
| pixel_inspection_results.zip | `/pixel_inspection_results.zip` | ZIP, 890 B | raster inventory CSVs | Downloaded + extracted ✓ |
| Vijayawada_Training_Data.zip | `/Vijayawada_Training_Data.zip` | ZIP, 35.6 MB | Member 1 training data | Downloaded + extracted ✓ |

## 2. `/GeoRAG_Satellite` (9 items)

| File | Exact path | File type | Purpose | Verified status |
|---|---|---|---|---|
| Vijayawada_Landsat_2015.tif | `/GeoRAG_Satellite/Vijayawada_Landsat_2015.tif` | GeoTIFF 21.5 MB | 7-band scene 2015, 693×669, 30 m, EPSG:4326 | Downloaded, opened, bands/statistics read ✓ |
| Vijayawada_Landsat_2020.tif | `/GeoRAG_Satellite/Vijayawada_Landsat_2020.tif` | GeoTIFF 20.8 MB | 7-band scene 2020, same grid | Downloaded, opened, read ✓ |
| 2025 | `/GeoRAG_Satellite/2025` | GeoTIFF (no extension) 44.5 MB | 7-band scene 2025, 743×743, shifted origin | Downloaded, opened, read ✓ |
| Vijayawada_2025_ML_Features.tif | `/GeoRAG_Satellite/Vijayawada_2025_ML_Features.tif` | GeoTIFF 32.3 MB | 10-band (7 raw + NDVI/NDBI/NDWI) 2025 | Downloaded, opened; index bands match formulas exactly ✓ |
| 2015.tif | `/GeoRAG_Satellite/2015.tif` | GeoTIFF 7.6 MB | 7-band subset scene 2015, 362×392 | Downloaded, opened, read ✓ |
| 2020.tif | `/GeoRAG_Satellite/2020.tif` | GeoTIFF 7.6 MB | 7-band subset scene 2020, 362×392 (9,779 NaN) | Downloaded, opened, read ✓ |
| 2015.tif.aux.xml | `/GeoRAG_Satellite/2015.tif.aux.xml` | XML 839 KB | GDAL statistics sidecar | Downloaded ✓ |
| 2020.tif.aux.xml | `/GeoRAG_Satellite/2020.tif.aux.xml` | XML 839 KB | GDAL statistics sidecar | Downloaded ✓ |
| Vijayawada_Boundary-SHP.zip | `/GeoRAG_Satellite/Vijayawada_Boundary-SHP.zip` | ZIP 5 KB | Vijayawada boundary shapefile | Downloaded, listed (7 files) ✓ |

Band order verified for all scenes: blue, green, red, nir, swir1, swir2, thermal.

## 3. `/GeoRAG_Indices` (3 items)

| File | Exact path | File type | Purpose | Verified status |
|---|---|---|---|---|
| Vijayawada_Indices_2015.tif | `/GeoRAG_Indices/Vijayawada_Indices_2015.tif` | GeoTIFF 4.9 MB | NDVI/NDBI/NDWI 2015, 693×669 | Downloaded, opened; matches recomputed formulas (max Δ 3e-8) ✓ |
| Vijayawada_Indices_2020.tif | `/GeoRAG_Indices/Vijayawada_Indices_2020.tif` | GeoTIFF 4.7 MB | NDVI/NDBI/NDWI 2020 | Same verification ✓ |
| Vijayawada_Indices_2025.tif | `/GeoRAG_Indices/Vijayawada_Indices_2025.tif` | GeoTIFF 3.7 MB | NDVI/NDBI/NDWI 2025 (older Sep-20 product) | Downloaded, opened; **199,388 NaN px; disagrees with the Oct-7 2025 scene** (see LIMITATIONS §3) ✓ |

## 4. `/GeoRAG_LandUse_Labels` (3 items)

| File | Exact path | File type | Purpose | Verified status |
|---|---|---|---|---|
| Vijayawada_LandUse_Labels_2015.tif | `/GeoRAG_LandUse_Labels/Vijayawada_LandUse_Labels_2015.tif` | GeoTIFF 118 KB | 10 m labels 2015 (2077×2004, values 0–4) | Downloaded, opened, class counts read ✓ |
| Vijayawada_LandUse_Labels_2020.tif | `/GeoRAG_LandUse_Labels/Vijayawada_LandUse_Labels_2020.tif` | GeoTIFF 126 KB | 10 m labels 2020 | Downloaded, opened ✓ |
| Vijayawada_LandUse_Labels_2025.tif | `/GeoRAG_LandUse_Labels/Vijayawada_LandUse_Labels_2025.tif` | GeoTIFF 126 KB | 10 m labels 2025 | Downloaded, opened ✓ |

Class counts match the 30 m aligned labels ×9 (within rounding) → same class scheme
(1 Water, 2 Urban, 3 Soil, 4 Vegetation).

## 5. `Vijayawada_Training_Data.zip` contents (extracted to `extracted/`)

| File | Path in ZIP | Type | Purpose | Verified status |
|---|---|---|---|---|
| README.txt | `Vijayawada_Training_Data/README.txt` | text | dataset description, class mapping | Read in full ✓ |
| Class_Count_Report.csv | `Vijayawada_Training_Data/Class_Count_Report.csv` | CSV | per-year class counts | Read; matches CSV counts ✓ |
| Training_Samples.csv | `Vijayawada_Training_Data/Training_Samples.csv` | CSV 113 MB | 956,134 rows: year,row,col,lon,lat,NDVI,NDBI,NDWI,class_id,class_name | Downloaded; header/counts/class mapping verified ✓ (only **3** index features — not the 9-feature set) |
| Labels_2015_Aligned.tif | `.../Labels_2015_Aligned.tif` | GeoTIFF 27 KB | 30 m labels 2015 on the shared grid | Opened; 693×669, transform matches Landsat exactly ✓ |
| Labels_2020_Aligned.tif | `.../Labels_2020_Aligned.tif` | GeoTIFF 28 KB | 30 m labels 2020 | Opened ✓ |
| Labels_2025_Aligned.tif | `.../Labels_2025_Aligned.tif` | GeoTIFF 29 KB | 30 m labels 2025 | Opened ✓ |

## 6. `pixel_inspection_results.zip` contents (extracted to `extracted/`)

| File | Type | Purpose | Verified status |
|---|---|---|---|
| pixel_inspection_results/raster_inventory.csv | CSV 462 B | raster inventory (2015/2020/2025: 7 bands, 362×392, EPSG:4326, valid-pixel counts) | Read ✓ |
| pixel_inspection_results/possible_class_counts.csv | CSV 2 B | (empty result) | Read ✓ |

## 7. Expected Member 2 artifacts — SEARCHED FOR AND **NOT FOUND** in the Drive

Verified absent after the complete recursive listing above:

- any Python / R / notebook source code of any kind
- Random Forest or XGBoost code, serialized models, or metrics files
- `training_2015.csv`, `training_2020.csv`, `training_2025.csv`,
  `training_all_years.csv` (the only training file is `Training_Samples.csv`)
- an extracted project folder named `georag`
- prediction GeoTIFFs, change-detection rasters, transition matrices, area statistics,
  validation results, feature-importance outputs, figures, README/reports,
  Member 2 handoff documents, or project archives other than the two ZIPs above

Consequence: previous Member 2 results cannot be compared against; RF/XGBoost were
re-trained under one shared protocol and are labelled as retrained baselines
(see `docs/LIMITATIONS.md` §4).

# {{TITLE}}

## Dataset descriptor

This package is release version **{{VERSION}}**, assembled without changing the Parquet bytes from the completed Gold layer (scientific-configuration fingerprint `{{FINGERPRINT}}`). The package version identifies this distribution; it does not imply that the pipeline was rerun.

The dataset contains aggregated, point-sampled ambulance operating speeds assigned to OpenStreetMap-derived, directed road features in Kraków and the surrounding processing area, Poland, during 2021–2023. Its six Parquet files represent {{TOTAL}} retained speed observations: {{OFFICIAL}} in the official Kraków bounding-box scope and {{EXTENDED}} in the extended processing-area scope. The official bounding box is not identical to the municipal polygon. For a Kraków-specific analysis, filter `spatial_scope` to `OFFICIAL_KRAKOW_BOUNDING_BOX`; do not pool both scopes without describing the enlarged study area.

These values are sampled speeds at telemetry points, not complete road-traversal times, a census of ambulance journeys, or a representative sample of general traffic. The release excludes raw trajectories, patient and dispatch records, exact observation timestamps, and source identifiers. It does contain aggregate counts of distinct source identifiers; those are not necessarily vehicle or trip counts.

## Files and relationships

| File | Rows |
|---|---:|
{{TABLE_ROWS}}

`krakow_roads_base.parquet` holds road-feature geometry (WGS84/EPSG:4326), OSM road attributes, and reference-speed provenance. Join any of the four temporal profile tables to it on `road_feature_id`. Those four tables are alternative aggregations of the **same** retained observations; their `n_samples` totals must not be added together. The flat table contains only observed combinations of temporal, weather, direction, and signal-state categories. Missing combinations are not zero-valued records.

All six files are Parquet; the road base and flat table contain GeoParquet geometry. Use a GeoParquet-aware reader for mapping. No CSV, GeoPackage, figures, raw input, or validation reports are included in this minimal deposit.

## Methods and variable interpretation

Input telemetry was screened for usable coordinates, identifiers, timestamps, and speed. `GPS_TIME`, interpreted as UTC, is the source of the published calendar and local time-of-day categories; `ETL_CZAS` is a local processing timestamp used for audit, not as a substitute. The provider `GPS_SPEED` field is in centimetres per second and was converted to km/h by multiplying by 0.036. The published speed includes stopped and near-stopped observations. Running speed includes only observations above the configured running threshold. Missing, non-finite, and negative provider speeds were excluded, with no silent displacement-speed fallback.

Road features were derived from a pinned 2023 OpenStreetMap snapshot. Candidate road segments were sought within 40 m of each eligible point; the primary Gold aggregation uses assignments within 15 m. Candidates were ranked using distance and, when available, trajectory/road alignment. A road-level upper-IQR rule removed flagged high-speed observations. Because map matching is inferential, a recorded road assignment is not proof that the vehicle travelled along that exact segment.

The `operating_speed_reference_kmh` field uses parsed OSM `maxspeed` tags where possible and otherwise a default inferred from the OSM `highway` type. Check `operating_speed_reference_source`. The reference is not a verified legal speed limit for every segment; `delta_operating_speed_reference_kmh` is a numerical comparison, not a speeding determination.

Common profile measures are `avg_travel_speed_kmh`, `avg_running_speed_kmh`, `median_speed_kmh`, `p85_speed_kmh`, `stddev_speed_kmh`, `stopped_observation_ratio`, `n_samples`, `n_unique_source_ids`, `n_unique_observation_days`, and `meets_min_sample_count`. The last flag only indicates whether the row meets its recorded `min_sample_count`; it does not establish independence, representativeness, or reliability. The `spatial_scope` field refers to the observation coordinate, not whether an entire road geometry lies within a boundary.

Season, calendar month, ISO weekday, and local time-of-day profiles are separate views. Time-of-day bands are `PEAK_MORNING` (07:00–09:59), `OFF_PEAK_DAY` (10:00–14:59), `PEAK_EVENING` (15:00–18:59), and `NIGHT` (remaining hours), in Europe/Warsaw local time. Calendar/month profiles pool available years; they do not contain a year field. Source coverage is especially sparse in January 2023 and absent from February through May 2023, so month/season differences must not be treated as balanced annual effects.

Direction categories are relative to the digitized OSM geometry, not legal travel permissions. Signal-state categories represent recorded light/sound combinations and do not establish mission purpose or statutory priority. Weather categories are derived from one ERA5/Open-Meteo reanalysis grid cell; they are not road-surface measurements. Unknown weather is retained explicitly.

## Suggested use

```python
import geopandas as gpd
import pandas as pd

roads = gpd.read_parquet("krakow_roads_base.parquet")
profile = pd.read_parquet("speeds_by_time_of_day.parquet")
profile = profile.loc[
    (profile["spatial_scope"] == "OFFICIAL_KRAKOW_BOUNDING_BOX")
    & profile["meets_min_sample_count"]
]
mapped = roads.merge(profile, on="road_feature_id", how="inner")
```

Analyses should also examine sample count, distinct source-ID count, observation-day count, road class, temporal coverage, and map-matching uncertainty. OSM-derived road geometry and attributes retain OpenStreetMap attribution and ODbL terms. Weather data are derived from Open-Meteo Historical Weather API / ERA5. See `LICENCE` for approved component terms and third-party notices.

## Citation and integrity

Use `citations.cff` for dataset citation. `checksums.sha256` contains SHA-256 digests for **all nine other files** in this directory and intentionally cannot checksum itself. Verify with `shasum -a 256 -c checksums.sha256` on macOS or `sha256sum -c checksums.sha256` where available. The package contains exactly ten top-level files and no subdirectories.

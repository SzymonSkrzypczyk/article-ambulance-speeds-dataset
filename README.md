# Kraków ambulance operating-speed dataset pipeline

[![DOI](https://zenodo.org/badge/1392437376.svg)](https://doi.org/10.5281/zenodo.23016931)

This repository contains the custom code used to generate, validate,
visualise, test, and export the lean public dataset of ambulance operating
speeds on road features in Kraków and its extended processing area.

## Setup

Create the pinned environment:

```bash
mamba env create -f environment.yml
conda activate karetki-publication
```

The environment pins Python 3.11, Java 11, PySpark 3.4.1, Apache Sedona 1.5.1,
and the geospatial and test dependencies used by the workflow.

## Inputs

Running the processing pipeline requires authorised source GPS CSV/ZIP files.
Those restricted records are not included in this repository. The pipeline
also retrieves the configured OpenStreetMap and weather inputs, recording
their provenance in the output directory.

## Generate the Gold dataset

Run the complete Bronze, Silver, and Gold workflow with an explicit dataset
version and source-identifier semantic:

```bash
python main_pipeline.py \
  --stage all \
  --force-rebuild \
  --dataset-version 1.0.0 \
  --source-id-semantic UNVERIFIED_SOURCE_IDENTIFIER
```

Pass the input/output paths and any documented scientific parameters required
for a particular run to `main_pipeline.py`.

## Validate and create figures

Run the validation and report commands against a completed output directory:

```bash
python validate_dataset.py --gold-dir /path/to/output/gold
python validate_map_matching.py \
  --silver-dir /path/to/output/silver \
  --reports-dir /path/to/output/reports
python summarize_silver_road_distances.py \
  --silver-dir /path/to/output/silver \
  --output-dir /path/to/output/reports/silver_road_distance_summary
python make_coverage_diagnostics_figure.py \
  --reports-dir /path/to/output/reports \
  --output /path/to/output/reports/coverage_diagnostics.png
python make_article_figures.py \
  --gold-dir /path/to/output/gold \
  --reports-dir /path/to/output/reports \
  --output-dir /path/to/output/reports/article_figures
```

## Export the lean public dataset

Copy `final_dataset_metadata.example.json` to a private metadata file, replace
its licence placeholder with approved component terms, then run:

```bash
python create_final_dataset.py \
  --gold-dir /path/to/output/gold \
  --metadata /path/to/approved_metadata.json \
  --output /path/to/new/final_dataset
```

The command creates exactly six canonical Parquet files, `README.md`,
`LICENCE`, `citations.cff`, and `checksums.sha256`.

## Tests

Run the unit tests:

```bash
python -m pytest
```

Spark/Sedona integration tests are opt-in because their first run may download
pinned JVM artifacts:

```bash
KARETKI_RUN_SPARK_TESTS=1 python -m pytest tests/test_spark_transforms.py
```

## Citation

```bibtex
@software{krakow_ambulance_operating_speed_dataset_pipeline,
  author  = {Skrzypczyk, Szymon and Lupa, Michał},
  title   = {Kraków ambulance operating-speed dataset pipeline},
  url     = {https://github.com/SzymonSkrzypczyk/article-ambulance-speeds-dataset},
  version = {1.0.0}
}
```

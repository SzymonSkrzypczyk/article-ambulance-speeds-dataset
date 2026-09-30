"""Pinned Open-Meteo reanalysis ingestion with immutable request provenance."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional, Tuple

import httpx
import pandas as pd

from config import config
from pipeline_utils import (
    layer_is_complete,
    sha256_file,
    utc_now_iso,
    write_json_atomic,
    write_layer_completion_marker,
)

logger = logging.getLogger(__name__)

SNOW_CODES = {66, 67, 71, 73, 75, 77, 85, 86}
RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 80, 81, 82, 95, 96, 99}
FOG_CODES = {45, 48}
DRY_CODES = {0, 1, 2, 3}


def map_wmo_code_to_category(
    weather_code: Optional[int],
    precipitation: Optional[float],
    snowfall: Optional[float],
) -> str:
    """Map reanalysis variables to meteorological categories without imputing missingness."""
    code = (
        int(weather_code)
        if weather_code is not None and not pd.isna(weather_code)
        else None
    )
    precip = (
        float(precipitation)
        if precipitation is not None and not pd.isna(precipitation)
        else None
    )
    snow = float(snowfall) if snowfall is not None and not pd.isna(snowfall) else None
    if code is None and precip is None and snow is None:
        return "UNKNOWN_WEATHER"
    if (snow is not None and snow > 0) or code in SNOW_CODES:
        return "SNOWFALL_OR_FREEZING_PRECIPITATION"
    if (precip is not None and precip > 0) or code in RAIN_CODES:
        return "RAIN_OR_LIQUID_PRECIPITATION"
    if code in FOG_CODES:
        return "FOG_OR_LOW_VISIBILITY"
    if (
        code in DRY_CODES
        and (precip is None or precip == 0)
        and (snow is None or snow == 0)
    ):
        return "NO_RECORDED_PRECIPITATION"
    return "OTHER_OR_UNCLASSIFIED_WEATHER"


def open_meteo_request_parameters(
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
) -> dict:
    """Return all API parameters explicitly so defaults cannot drift silently."""
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation,rain,snowfall,weather_code,wind_speed_10m",
        "models": config.WEATHER_MODEL,
        "timezone": config.WEATHER_TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
        "cell_selection": "nearest",
    }


def _response_to_frame(
    data: dict,
    requested_latitude: float = config.KRAKOW_LAT,
    requested_longitude: float = config.KRAKOW_LON,
) -> pd.DataFrame:
    if "hourly" not in data or "time" not in data["hourly"]:
        raise ValueError("Open-Meteo response does not contain hourly time-series data")
    hourly = data["hourly"]
    required = [
        "time",
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "wind_speed_10m",
    ]
    missing = [name for name in required if name not in hourly]
    if missing:
        raise ValueError(f"Open-Meteo response is missing variables: {missing}")
    lengths = {name: len(hourly[name]) for name in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Open-Meteo arrays have inconsistent lengths: {lengths}")

    frame = pd.DataFrame({name: hourly[name] for name in required})
    parsed_time = pd.to_datetime(frame.pop("time"), utc=True, errors="raise")
    frame.insert(0, "weather_hour_utc", parsed_time.dt.tz_localize(None))
    if frame["weather_hour_utc"].duplicated().any():
        raise ValueError("Open-Meteo response contains duplicate hourly timestamps")
    expected = pd.date_range(
        frame["weather_hour_utc"].min(),
        frame["weather_hour_utc"].max(),
        freq="h",
    )
    missing_hours = expected.difference(pd.DatetimeIndex(frame["weather_hour_utc"]))
    if len(missing_hours):
        raise ValueError(
            f"Open-Meteo response has {len(missing_hours)} missing hourly timestamps"
        )

    frame["weather_category"] = [
        map_wmo_code_to_category(code, precipitation, snowfall)
        for code, precipitation, snowfall in zip(
            frame["weather_code"], frame["precipitation"], frame["snowfall"]
        )
    ]
    frame["weather_model"] = config.WEATHER_MODEL
    frame["requested_latitude"] = requested_latitude
    frame["requested_longitude"] = requested_longitude
    frame["grid_latitude"] = data.get("latitude")
    frame["grid_longitude"] = data.get("longitude")
    frame["grid_elevation_m"] = data.get("elevation")
    return frame


def fetch_open_meteo_weather(
    start_date: str = config.EXPECTED_START_DATE,
    end_date: str = config.EXPECTED_END_DATE,
    lat: float = config.KRAKOW_LAT,
    lon: float = config.KRAKOW_LON,
    client: Optional[httpx.Client] = None,
) -> Tuple[pd.DataFrame, dict, dict]:
    """Fetch a pinned reanalysis product and return frame, raw response, and request."""
    params = open_meteo_request_parameters(start_date, end_date, lat, lon)
    logger.info(
        "Fetching Open-Meteo model=%s for %s through %s",
        config.WEATHER_MODEL,
        start_date,
        end_date,
    )
    owns_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=httpx.Timeout(120.0, connect=30.0), follow_redirects=True
        )
    try:
        response = client.get(config.WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            client.close()
    if data.get("error"):
        raise RuntimeError(f"Open-Meteo error: {data.get('reason', 'unknown reason')}")
    return _response_to_frame(data, lat, lon), data, params


def _write_weather_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write Spark-3.4-compatible timestamps instead of Parquet nanoseconds."""
    frame.to_parquet(
        path,
        index=False,
        engine="pyarrow",
        coerce_timestamps="us",
        allow_truncated_timestamps=False,
    )


def process_weather_data(force_rebuild: bool = False) -> Path:
    """Create weather Parquet plus raw-response and provenance JSON artifacts."""
    outputs_exist = (
        config.WEATHER_PARQUET_PATH.exists()
        and config.WEATHER_RAW_RESPONSE_PATH.exists()
        and config.WEATHER_PROVENANCE_PATH.exists()
    )
    if outputs_exist and layer_is_complete(
        config.WEATHER_DIR,
        layer="weather",
        configuration_fingerprint=config.configuration_fingerprint(),
        required_artifacts=(
            config.WEATHER_PARQUET_PATH.relative_to(config.WEATHER_DIR),
            config.WEATHER_RAW_RESPONSE_PATH.relative_to(config.WEATHER_DIR),
            config.WEATHER_PROVENANCE_PATH.relative_to(config.WEATHER_DIR),
        ),
    ) and not force_rebuild:
        logger.info("Weather outputs already exist; skipping fetch")
        return config.WEATHER_PARQUET_PATH

    config.ensure_directories()
    retrieved_at = utc_now_iso()
    # The study period is defined on Warsaw-local dates but the request is
    # interpreted in UTC
    fetch_start = (
        date.fromisoformat(config.EXPECTED_START_DATE) - timedelta(days=1)
    ).isoformat()
    frame, raw_response, params = fetch_open_meteo_weather(
        start_date=fetch_start,
        end_date=config.EXPECTED_END_DATE,
    )
    write_json_atomic(config.WEATHER_RAW_RESPONSE_PATH, raw_response)
    _write_weather_parquet(frame, config.WEATHER_PARQUET_PATH)

    category_counts = {
        str(key): int(value)
        for key, value in frame["weather_category"].value_counts().items()
    }
    provenance: dict[str, Any] = {
        "provider": "Open-Meteo Historical Weather API",
        "api_url": config.WEATHER_API_URL,
        "request_parameters": params,
        "study_period_dates_warsaw": {
            "start": config.EXPECTED_START_DATE,
            "end": config.EXPECTED_END_DATE,
        },
        "retrieved_at_utc": retrieved_at,
        "model": config.WEATHER_MODEL,
        "spatial_representation": config.WEATHER_LOCATION_DESCRIPTION,
        "requested_coordinate_wgs84": [config.KRAKOW_LAT, config.KRAKOW_LON],
        "returned_grid_coordinate_wgs84": [
            raw_response.get("latitude"),
            raw_response.get("longitude"),
        ],
        "returned_elevation_m": raw_response.get("elevation"),
        "returned_timezone": raw_response.get("timezone"),
        "returned_utc_offset_seconds": raw_response.get("utc_offset_seconds"),
        "hourly_units": raw_response.get("hourly_units"),
        "row_count": int(len(frame)),
        "actual_start_utc": frame["weather_hour_utc"].min().isoformat(),
        "actual_end_utc": frame["weather_hour_utc"].max().isoformat(),
        "missing_values_by_column": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "category_counts": category_counts,
        "raw_response_sha256": sha256_file(config.WEATHER_RAW_RESPONSE_PATH),
        "weather_parquet_sha256": sha256_file(config.WEATHER_PARQUET_PATH),
    }
    write_json_atomic(config.WEATHER_PROVENANCE_PATH, provenance)
    write_layer_completion_marker(
        config.WEATHER_DIR,
        {
            "layer": "weather",
            "dataset_version": config.DATASET_VERSION,
            "artifacts": [
                str(config.WEATHER_PARQUET_PATH.relative_to(config.WEATHER_DIR)),
                str(config.WEATHER_RAW_RESPONSE_PATH.relative_to(config.WEATHER_DIR)),
                str(config.WEATHER_PROVENANCE_PATH.relative_to(config.WEATHER_DIR)),
            ],
            "configuration_fingerprint": config.configuration_fingerprint(),
        },
    )
    logger.info("Saved %d pinned hourly weather records", len(frame))
    return config.WEATHER_PARQUET_PATH


if __name__ == "__main__":
    process_weather_data(force_rebuild=True)

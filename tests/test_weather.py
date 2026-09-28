import pandas as pd
import pyarrow.parquet as pq
import pytest

from etl.weather import (
    _response_to_frame,
    _write_weather_parquet,
    map_wmo_code_to_category,
    open_meteo_request_parameters,
)


def test_weather_parquet_uses_spark_compatible_timestamp_units(tmp_path):
    path = tmp_path / "weather.parquet"
    frame = pd.DataFrame(
        {"weather_hour_utc": [pd.Timestamp("2021-01-01T00:00:00")]}
    )
    _write_weather_parquet(frame, path)
    assert pq.read_schema(path).field("weather_hour_utc").type.unit == "us"


def test_missing_weather_is_not_dry():
    assert map_wmo_code_to_category(None, None, None) == "UNKNOWN_WEATHER"


@pytest.mark.parametrize(
    ("code", "precipitation", "snowfall", "expected"),
    [
        (0, 0.0, 0.0, "NO_RECORDED_PRECIPITATION"),
        (61, 0.0, 0.0, "RAIN_OR_LIQUID_PRECIPITATION"),
        (71, 0.0, 0.0, "SNOWFALL_OR_FREEZING_PRECIPITATION"),
        (45, 0.0, 0.0, "FOG_OR_LOW_VISIBILITY"),
    ],
)
def test_wmo_categories(code, precipitation, snowfall, expected):
    assert map_wmo_code_to_category(code, precipitation, snowfall) == expected


def test_request_pins_model_and_units():
    params = open_meteo_request_parameters("2021-01-01", "2021-01-02", 50.0, 20.0)
    assert params["models"]
    assert params["timezone"] == "UTC"
    assert params["wind_speed_unit"] == "kmh"


def test_response_rejects_missing_hours():
    response = {
        "hourly": {
            "time": ["2021-01-01T00:00", "2021-01-01T02:00"],
            "temperature_2m": [0.0, 0.0],
            "precipitation": [0.0, 0.0],
            "rain": [0.0, 0.0],
            "snowfall": [0.0, 0.0],
            "weather_code": [0, 0],
            "wind_speed_10m": [1.0, 1.0],
        }
    }
    with pytest.raises(ValueError, match="missing hourly"):
        _response_to_frame(response)


def test_response_preserves_overridden_requested_coordinate():
    response = {
        "latitude": 50.1,
        "longitude": 20.1,
        "hourly": {
            "time": ["2021-01-01T00:00"],
            "temperature_2m": [0.0],
            "precipitation": [0.0],
            "rain": [0.0],
            "snowfall": [0.0],
            "weather_code": [0],
            "wind_speed_10m": [1.0],
        },
    }
    frame = _response_to_frame(response, 49.9, 19.9)
    assert frame.loc[0, "requested_latitude"] == 49.9
    assert frame.loc[0, "requested_longitude"] == 19.9

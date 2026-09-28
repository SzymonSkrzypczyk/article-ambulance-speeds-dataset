import math

import pandas as pd

from validate_map_matching import (
    _lowest_point_ids,
    add_manual_ground_truth_metrics,
    metric_rows,
)


def test_lowest_point_ids_uses_stable_lexical_order_for_hashes():
    frame = pd.DataFrame(
        {
            "point_id": ["f0", "0a", "10", "0a"],
            "row": [1, 2, 3, 4],
        }
    )

    result = _lowest_point_ids(frame, 3)

    assert result["point_id"].tolist() == ["0a", "0a", "10"]
    assert result["row"].tolist() == [2, 4, 3]


def test_validation_detects_invariant_violations():
    sample = pd.DataFrame(
        {
            "year": [2021, 2021],
            "highway": ["residential", "residential"],
            "dist_to_axis": [2.0, 16.0],
            "heading_alignment_diff": [0.1, math.pi],
            "azimuth_diff": [0.2, 4.0],
            "candidate_count": [1, 0],
            "within_primary_match_radius": [True, False],
            "score_margin": [None, -0.2],
            "dist_margin_m": [None, -1.0],
            "weather_join_status": ["MATCHED", "MISSING_WEATHER_HOUR"],
            "is_high_speed_iqr_outlier": [False, True],
            "calc_speed_kmh": [30.0, 90.0],
        }
    )
    metrics = {row["metric"]: row["value"] for row in metric_rows(sample)}
    assert metrics["invariant_violation_total"] >= 4
    assert metrics["missing_weather_join"] == 1


def test_validation_accepts_valid_ranges():
    sample = pd.DataFrame(
        {
            "year": [2021],
            "highway": ["primary"],
            "dist_to_axis": [2.0],
            "heading_alignment_diff": [0.1],
            "azimuth_diff": [0.2],
            "candidate_count": [1],
            "within_primary_match_radius": [True],
            "score_margin": [None],
            "dist_margin_m": [None],
            "weather_join_status": ["MATCHED"],
            "is_high_speed_iqr_outlier": [False],
            "calc_speed_kmh": [30.0],
        }
    )
    metrics = {row["metric"]: row["value"] for row in metric_rows(sample)}
    assert metrics["invariant_violation_total"] == 0


def test_textual_false_manual_label_is_false(tmp_path):
    sample = pd.DataFrame({"point_id": ["a", "b"]})
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "point_id,correct_road_match\na,False\nb,True\n", encoding="utf-8"
    )
    metrics, reviewed = add_manual_ground_truth_metrics([], sample, labels)
    accuracy = next(
        row["value"] for row in metrics if row["metric"] == "manual_match_accuracy"
    )
    assert accuracy == 50.0
    assert reviewed is not None


def test_blank_manual_template_rows_are_ignored(tmp_path):
    sample = pd.DataFrame({"point_id": ["a", "b"]})
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "point_id,correct_road_match\na,\nb,True\n", encoding="utf-8"
    )

    metrics, reviewed = add_manual_ground_truth_metrics([], sample, labels)

    accuracy = next(
        row["value"] for row in metrics if row["metric"] == "manual_match_accuracy"
    )
    assert accuracy == 100.0
    assert reviewed is not None
    assert reviewed["point_id"].tolist() == ["b"]


def test_empty_manual_template_records_no_completed_labels(tmp_path):
    sample = pd.DataFrame({"point_id": ["a"]})
    labels = tmp_path / "labels.csv"
    labels.write_text("point_id,correct_road_match\na,\n", encoding="utf-8")

    metrics, reviewed = add_manual_ground_truth_metrics([], sample, labels)

    assert metrics == [
        {
            "metric": "manual_ground_truth_status",
            "value": "NO_COMPLETED_LABELS",
            "unit": "status",
            "scope": "manual_ground_truth",
        }
    ]
    assert reviewed is None

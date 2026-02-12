"""Tests for the zone analyzer — point-in-polygon and observation bridge."""

import pytest

from app.cv.zone_analyzer import ZoneAnalyzer
from app.schemas import BoundingBox, Detection, ZoneDefinition


def _make_detection(x: float, y: float, label: str = "car") -> Detection:
    return Detection(
        detection_id="det_test",
        label=label,
        confidence=0.9,
        bbox=BoundingBox(x1=x - 20, y1=y - 40, x2=x + 20, y2=y),
        center_x=x,
        center_y=y,
    )


@pytest.fixture
def analyzer() -> ZoneAnalyzer:
    return ZoneAnalyzer()


def test_detection_inside_zone(analyzer: ZoneAnalyzer) -> None:
    zone = ZoneDefinition(
        zone_id="z1",
        zone_type="parking",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    analyzer.load_zones([zone])

    det = _make_detection(50, 50)
    results = analyzer.assign_detections_to_zones([det])

    assert len(results) == 1
    assert results[0].zone is not None
    assert results[0].zone.zone_id == "z1"
    assert results[0].lane_type == "parking"


def test_detection_outside_zone(analyzer: ZoneAnalyzer) -> None:
    zone = ZoneDefinition(
        zone_id="z1",
        zone_type="parking",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    analyzer.load_zones([zone])

    det = _make_detection(200, 200)
    results = analyzer.assign_detections_to_zones([det])

    assert len(results) == 1
    assert results[0].zone is None
    assert results[0].lane_type == "unknown"


def test_vehicle_type_mapping(analyzer: ZoneAnalyzer) -> None:
    zone = ZoneDefinition(
        zone_id="z1",
        zone_type="bus_lane",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    analyzer.load_zones([zone])

    results = analyzer.assign_detections_to_zones([
        _make_detection(50, 50, "car"),
        _make_detection(50, 50, "truck"),
        _make_detection(50, 50, "bus"),
        _make_detection(50, 50, "bicycle"),
    ])

    assert results[0].vehicle_type == "passenger"
    assert results[1].vehicle_type == "commercial"
    assert results[2].vehicle_type == "bus"
    assert results[3].vehicle_type == "bike"


def test_detections_to_observations(analyzer: ZoneAnalyzer) -> None:
    zone = ZoneDefinition(
        zone_id="z1",
        zone_type="double_parking",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    analyzer.load_zones([zone])

    assignments = analyzer.assign_detections_to_zones([_make_detection(50, 50)])
    observations = analyzer.detections_to_observations(assignments)

    assert len(observations) == 1
    obs = observations[0]
    assert obs.is_double_parked is True
    assert obs.vehicle_type == "passenger"
    assert obs.lane_type == "travel"


def test_bus_lane_obstruction(analyzer: ZoneAnalyzer) -> None:
    zone = ZoneDefinition(
        zone_id="z1",
        zone_type="bus_lane",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    analyzer.load_zones([zone])

    # A car in a bus lane should be flagged as obstructing
    assignments = analyzer.assign_detections_to_zones([_make_detection(50, 50, "car")])
    observations = analyzer.detections_to_observations(assignments)
    assert observations[0].is_obstructing is True

    # A bus in a bus lane should NOT be obstructing
    assignments = analyzer.assign_detections_to_zones([_make_detection(50, 50, "bus")])
    observations = analyzer.detections_to_observations(assignments)
    assert observations[0].is_obstructing is False


def test_zone_priority_first_wins(analyzer: ZoneAnalyzer) -> None:
    """When zones overlap, the first loaded zone wins."""
    z1 = ZoneDefinition(
        zone_id="z1", zone_type="parking",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    z2 = ZoneDefinition(
        zone_id="z2", zone_type="no_parking",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    )
    analyzer.load_zones([z1, z2])

    results = analyzer.assign_detections_to_zones([_make_detection(50, 50)])
    assert results[0].zone is not None
    assert results[0].zone.zone_id == "z1"

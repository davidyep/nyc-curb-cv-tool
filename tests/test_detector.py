"""Tests for the YOLOv8 vehicle detector wrapper."""

import numpy as np
import pytest

from app.cv.detector import VehicleDetector, _VEHICLE_CLASSES
from app.schemas import Detection


@pytest.fixture(scope="module")
def detector() -> VehicleDetector:
    return VehicleDetector(model_name="yolov8n.pt", confidence_threshold=0.25)


def test_detect_returns_list(detector: VehicleDetector) -> None:
    # Blank 640x480 image — expect no detections
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(blank)
    assert isinstance(results, list)
    for det in results:
        assert isinstance(det, Detection)


def test_detection_fields(detector: VehicleDetector) -> None:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(blank)
    for det in results:
        assert det.label in _VEHICLE_CLASSES.values()
        assert 0 <= det.confidence <= 1.0
        assert det.bbox.x1 <= det.bbox.x2
        assert det.bbox.y1 <= det.bbox.y2
        assert det.center_y == det.bbox.y2  # bottom-center


def test_vehicle_classes_only(detector: VehicleDetector) -> None:
    """Detections should only contain vehicle class labels."""
    valid_labels = set(_VEHICLE_CLASSES.values())
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for det in detector.detect(blank):
        assert det.label in valid_labels

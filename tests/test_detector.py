"""Tests for the YOLOv8 vehicle detector wrapper."""

import numpy as np
import pytest

from app.cv.detector import VehicleDetector, _DETECT_CLASSES
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
        assert det.label in _DETECT_CLASSES.values()
        assert 0 <= det.confidence <= 1.0
        assert det.bbox.x1 <= det.bbox.x2
        assert det.bbox.y1 <= det.bbox.y2
        assert det.center_y == det.bbox.y2  # bottom-center
        assert det.classification in ("commercial", "private", "unknown")
        assert isinstance(det.is_stationary, bool)


def test_detect_classes_only(detector: VehicleDetector) -> None:
    """Detections should only contain labels from _DETECT_CLASSES."""
    valid_labels = set(_DETECT_CLASSES.values())
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for det in detector.detect(blank):
        assert det.label in valid_labels


def test_classification_heuristic(detector: VehicleDetector) -> None:
    """Verify the classification mapping is consistent."""
    from app.cv.detector import _COMMERCIAL_LABELS, _PRIVATE_LABELS

    assert "truck" in _COMMERCIAL_LABELS
    assert "bus" in _COMMERCIAL_LABELS
    assert "car" in _PRIVATE_LABELS
    assert "motorcycle" in _PRIVATE_LABELS

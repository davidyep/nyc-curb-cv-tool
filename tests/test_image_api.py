"""Integration tests for the /analyze/image and /detect endpoints."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_test_image_bytes() -> bytes:
    """Create a simple 200x200 test image as JPEG bytes."""
    import cv2

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw a rectangle to give the detector something (may not trigger detection)
    cv2.rectangle(img, (50, 50), (150, 150), (200, 200, 200), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health_still_works() -> None:
    """Regression: existing /health endpoint remains functional."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_detect_endpoint() -> None:
    img_bytes = _make_test_image_bytes()
    resp = client.post(
        "/detect",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_analyze_image_endpoint() -> None:
    img_bytes = _make_test_image_bytes()
    request_payload = {
        "frame": {
            "frame_id": "test_frame_001",
            "camera_id": "cam_test",
            "timestamp_utc": "2026-01-15T10:00:00Z",
            "borough": "manhattan",
            "segment_id": "seg_test",
        },
        "zones": [
            {
                "zone_id": "z_test",
                "zone_type": "parking",
                "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]],
                "label": "Test Parking",
            }
        ],
    }

    resp = client.post(
        "/analyze/image",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        data={"request_json": json.dumps(request_payload)},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["frame_id"] == "test_frame_001"
    assert "detections" in data
    assert "zone_assignments" in data
    assert "decisions" in data
    assert "summary" in data
    assert "occupancy_rate" in data
    assert data["image_width"] == 200
    assert data["image_height"] == 200


def test_analyze_image_no_zones() -> None:
    """Analysis with no zones should still work — all vehicles unassigned."""
    img_bytes = _make_test_image_bytes()
    request_payload = {
        "frame": {
            "frame_id": "test_frame_002",
            "camera_id": "cam_test",
            "timestamp_utc": "2026-01-15T10:00:00Z",
            "borough": "brooklyn",
            "segment_id": "seg_test",
        },
        "zones": [],
    }

    resp = client.post(
        "/analyze/image",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        data={"request_json": json.dumps(request_payload)},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["frame_id"] == "test_frame_002"
    # All zone_assignments should have zone=None
    for za in data.get("zone_assignments", []):
        assert za["zone"] is None

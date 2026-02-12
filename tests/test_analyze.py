from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_illegal_detection() -> None:
    payload = {
        "frame": {
            "frame_id": "f1",
            "camera_id": "cam_01",
            "timestamp_utc": datetime(2026, 1, 10, 1, 0, tzinfo=timezone.utc).isoformat(),
            "borough": "manhattan",
            "segment_id": "seg_1001",
        },
        "observations": [
            {
                "track_id": "t1",
                "vehicle_type": "commercial",
                "lane_type": "bus",
                "is_double_parked": True,
                "is_obstructing": False,
                "curb_distance_m": 0.5,
                "dwell_time_seconds": 2200,
            }
        ],
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["summary"]["likely_illegal"] == 1
    assert data["decisions"][0]["status"] == "likely_illegal"
    assert "double_parking_detected" in data["decisions"][0]["reason_codes"]

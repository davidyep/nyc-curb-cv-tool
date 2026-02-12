# NYC Curb Utilization Tool

Computer-vision-powered curb utilization analysis for NYC streets. Upload a street image, draw infrastructure zones (parking, bus lane, bike lane, etc.), and get instant vehicle detection with legal/illegal parking classification.

## Features

- **YOLOv8 Vehicle Detection** — detects cars, trucks, buses, motorcycles, and bicycles in street images
- **Interactive Zone Drawing** — draw polygon zones on uploaded images to define parking, bus lanes, bike lanes, no-parking zones, fire hydrant areas, and more
- **Automatic Lane Detection** — color-based segmentation auto-detects NYC bus lane (red) and bike lane (green) paint
- **NYC Parking Rules Engine** — evaluates legality based on zone type, vehicle type, dwell time, double parking, and overnight restrictions
- **Annotated Image Output** — bounding boxes color-coded by legality (green=legal, red=illegal, yellow=uncertain) with zone overlays
- **Three-Tab Dashboard** — upload & draw zones, view analysis results with charts, and browse historical analytics
- **REST API** — FastAPI endpoints for programmatic image analysis and detection

## Architecture

```
Image Upload + User-Drawn Zones (Streamlit Dashboard)
    │
    ▼
POST /analyze/image (FastAPI)
    │
    ├─► app/cv/detector.py ──────► YOLOv8 vehicle detections
    ├─► app/cv/lane_detector.py ─► auto-detect bus/bike lane paint
    ├─► app/cv/zone_analyzer.py ─► map detections → zones (Shapely)
    │       │
    │       ▼
    │   VehicleObservation objects
    │       │
    │       ▼
    ├─► app/rules_engine.py ─────► LegalityDecision (zone-aware)
    ├─► app/cv/annotator.py ─────► annotated image with overlays
    └─► app/analytics.py ────────► occupancy + violation metrics
            │
            ▼
    ImageAnalyzeResponse → Dashboard Results Tab
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. Start the API server

```bash
uvicorn app.main:app --reload
```

The first run will download the YOLOv8 nano model (~6 MB, cached automatically).

### 3. Start the dashboard

```bash
streamlit run dashboard/app.py
```

### 4. Analyze a street image

1. Open the dashboard (default: http://localhost:8501)
2. **Tab 1**: Upload a street image and select a zone type from the sidebar
3. Draw polygons on the image to define infrastructure zones
4. Click **Run Analysis**
5. **Tab 2**: View the annotated image, detection table, legality decisions, and charts

### 5. Run tests

```bash
pytest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/analyze` | POST | Analyze structured vehicle observations (original endpoint) |
| `/analyze/image` | POST | Analyze an uploaded image with zone definitions (multipart) |
| `/detect` | POST | Run vehicle detection only — no zones or legality |

### Example: `/analyze/image`

```bash
curl -X POST http://localhost:8000/analyze/image \
  -F "image=@street_photo.jpg" \
  -F 'request_json={"frame":{"frame_id":"f1","camera_id":"cam1","timestamp_utc":"2026-01-15T10:00:00Z","borough":"manhattan","segment_id":"seg_001"},"zones":[{"zone_id":"z1","zone_type":"parking","polygon":[[100,300],[500,300],[500,500],[100,500]]}]}'
```

## Zone Types

| Zone Type | Description | Legality |
|-----------|-------------|----------|
| `parking` | Legal parking zone | Standard rules apply |
| `no_parking` | No parking/standing at any time | Always illegal |
| `bus_lane` | Bus-only lane | Illegal for non-buses |
| `bike_lane` | Bike-only lane | Illegal for motor vehicles |
| `loading_zone` | Commercial loading zone | Commercial vehicles only |
| `fire_hydrant` | Within 15 ft of hydrant | Always illegal |
| `double_parking` | Double parking zone | Always illegal |
| `travel_lane` | Active travel lane | No standing |

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI endpoints
│   ├── schemas.py            # Pydantic data models
│   ├── rules_engine.py       # NYC parking rules evaluation
│   ├── analytics.py          # Occupancy and violation metrics
│   └── cv/
│       ├── detector.py       # YOLOv8 vehicle detection
│       ├── lane_detector.py  # HSV color-based lane detection
│       ├── zone_analyzer.py  # Shapely polygon zone assignment
│       └── annotator.py      # Image annotation drawing
├── config/
│   └── nyc_parking_rules.yaml
├── dashboard/
│   └── app.py                # Streamlit 3-tab dashboard
├── data/
│   └── sample_results.csv
├── tests/
│   ├── test_analyze.py       # Original API tests
│   ├── test_detector.py      # Vehicle detector tests
│   ├── test_zone_analyzer.py # Zone analyzer tests
│   └── test_image_api.py     # Image API integration tests
└── pyproject.toml
```

## Tech Stack

- **Computer Vision**: Ultralytics YOLOv8, OpenCV
- **Geometry**: Shapely (point-in-polygon zone assignment)
- **API**: FastAPI + Uvicorn
- **Dashboard**: Streamlit + streamlit-drawable-canvas + Plotly
- **Data Validation**: Pydantic v2
- **Rules Config**: YAML

## Roadmap

1. Video stream support with vehicle tracking (arrival → dwell → departure lifecycle)
2. Geospatial rules tied to blockface and posted signage/time windows
3. Custom model fine-tuned on NYC street imagery for higher accuracy
4. Map-native dashboard with corridor filtering and incident playback
5. Borough-, weather-, and lighting-specific model calibration

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

from app.analytics import occupancy_rate, summarize_decisions
from app.cv.annotator import draw_annotations
from app.cv.detector import VehicleDetector
from app.cv.lane_detector import LaneDetector
from app.cv.zone_analyzer import ZoneAnalyzer
from app.rules_engine import RulesEngine
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    Detection,
    ImageAnalyzeRequest,
    ImageAnalyzeResponse,
)

app = FastAPI(title="NYC Curb Utilization Tool", version="0.2.0")

# Shared singletons — loaded once at startup
rules_engine = RulesEngine()
vehicle_detector = VehicleDetector()
lane_detector = LaneDetector()
zone_analyzer = ZoneAnalyzer()


# ---------------------------------------------------------------------------
# Existing endpoints (unchanged)
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    decisions = [rules_engine.evaluate(request.frame, obs) for obs in request.observations]
    return AnalyzeResponse(
        frame_id=request.frame.frame_id,
        occupancy_rate=occupancy_rate(request.observations),
        decisions=decisions,
        summary=summarize_decisions(decisions),
    )


# ---------------------------------------------------------------------------
# New CV-powered endpoints
# ---------------------------------------------------------------------------


@app.post("/analyze/image", response_model=ImageAnalyzeResponse)
async def analyze_image(
    image: UploadFile = File(...),
    request_json: str = Form(...),
) -> ImageAnalyzeResponse:
    """Analyze an uploaded street image with user-defined zones.

    The image is sent as a file upload.  The request metadata (frame context
    and zone definitions) is sent as a JSON string in a form field because
    multipart forms cannot carry structured JSON alongside files.
    """
    # 1. Parse request
    request = ImageAnalyzeRequest.model_validate_json(request_json)

    # 2. Decode image
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = cv_image.shape[:2]

    # 3. Detect vehicles
    detections = vehicle_detector.detect(cv_image)

    # 4. Auto-detect bus/bike lanes from paint color
    auto_zones = lane_detector.detect_lanes(cv_image)

    # 5. Combine user zones (priority) with auto-detected zones
    all_zones = list(request.zones) + auto_zones
    zone_analyzer.load_zones(all_zones)

    # 6. Assign detections to zones
    assignments = zone_analyzer.assign_detections_to_zones(detections)

    # 7. Bridge to existing rules engine
    observations = zone_analyzer.detections_to_observations(assignments)
    decisions = [
        rules_engine.evaluate_with_zone(request.frame, obs, assignment.zone)
        for obs, assignment in zip(observations, assignments)
    ]

    # 8. Analytics
    occ = occupancy_rate(observations)
    summary = summarize_decisions(decisions)

    # 9. Annotate image
    annotated = draw_annotations(cv_image, detections, assignments, decisions, all_zones)
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return ImageAnalyzeResponse(
        frame_id=request.frame.frame_id,
        image_width=w,
        image_height=h,
        detections=detections,
        zone_assignments=assignments,
        occupancy_rate=occ,
        decisions=decisions,
        summary=summary,
        annotated_image_b64=annotated_b64,
    )


@app.post("/detect", response_model=list[Detection])
async def detect_only(image: UploadFile = File(...)) -> list[Detection]:
    """Run vehicle detection only — no zones or legality analysis."""
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return vehicle_detector.detect(cv_image)

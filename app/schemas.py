from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

VehicleType = Literal["passenger", "commercial", "bus", "bike", "scooter", "other"]
LaneType = Literal["travel", "bus", "bike", "parking", "unknown"]
ZoneType = Literal[
    "parking", "no_parking", "bus_lane", "bike_lane",
    "loading_zone", "fire_hydrant", "double_parking", "travel_lane",
]
DetectedClass = Literal[
    "car", "truck", "bus", "motorcycle", "bicycle", "person",
]
VehicleClassification = Literal["commercial", "private", "municipal", "unknown"]


class FrameContext(BaseModel):
    frame_id: str
    camera_id: str
    timestamp_utc: datetime
    borough: Literal["manhattan", "brooklyn", "queens", "bronx", "staten_island"]
    segment_id: str = Field(description="Unique curb segment identifier")


class VehicleObservation(BaseModel):
    track_id: str
    vehicle_type: VehicleType
    lane_type: LaneType
    is_double_parked: bool = False
    is_obstructing: bool = False
    curb_distance_m: float = 0.0
    dwell_time_seconds: int = 0


class AnalyzeRequest(BaseModel):
    frame: FrameContext
    observations: list[VehicleObservation]


class LegalityDecision(BaseModel):
    track_id: str
    status: Literal["legal", "likely_illegal", "uncertain", "in_transit"]
    reason_codes: list[str]
    confidence: float


class AnalyzeResponse(BaseModel):
    frame_id: str
    occupancy_rate: float
    decisions: list[LegalityDecision]
    summary: dict[str, int]


# ---------------------------------------------------------------------------
# CV-related models
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Pixel-space bounding box from detector."""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single object detection from YOLOv8."""
    detection_id: str
    label: DetectedClass
    confidence: float
    bbox: BoundingBox
    center_x: float
    center_y: float
    classification: VehicleClassification = "unknown"
    is_stationary: bool = True


class ZoneDefinition(BaseModel):
    """A polygon zone drawn by the user on the image."""
    zone_id: str
    zone_type: ZoneType
    polygon: list[tuple[float, float]]
    label: str = ""


class DetectionInZone(BaseModel):
    """A detection mapped to a zone."""
    detection: Detection
    zone: ZoneDefinition | None = None
    vehicle_type: VehicleType
    lane_type: LaneType
    is_in_transit: bool = False


class ImageAnalyzeRequest(BaseModel):
    """Request metadata for image-based analysis (image sent as file upload)."""
    frame: FrameContext
    zones: list[ZoneDefinition]


class ImageAnalyzeResponse(BaseModel):
    """Full response from image-based analysis."""
    frame_id: str
    image_width: int
    image_height: int
    detections: list[Detection]
    zone_assignments: list[DetectionInZone]
    occupancy_rate: float
    decisions: list[LegalityDecision]
    summary: dict[str, int]
    annotated_image_b64: str | None = None


# ---------------------------------------------------------------------------
# Mapping helpers: bridge between COCO labels / zone types and existing enums
# ---------------------------------------------------------------------------

COCO_TO_VEHICLE_TYPE: dict[str, VehicleType] = {
    "car": "passenger",
    "truck": "commercial",
    "bus": "bus",
    "motorcycle": "scooter",
    "bicycle": "bike",
    "person": "other",
}

# Labels that represent commercial-type vehicles
COMMERCIAL_LABELS: set[str] = {"truck", "bus"}
# Labels that represent private/personal vehicles
PRIVATE_LABELS: set[str] = {"car", "motorcycle"}

ZONE_TO_LANE_TYPE: dict[str, LaneType] = {
    "parking": "parking",
    "no_parking": "parking",
    "bus_lane": "bus",
    "bike_lane": "bike",
    "loading_zone": "parking",
    "fire_hydrant": "parking",
    "double_parking": "travel",
    "travel_lane": "travel",
}

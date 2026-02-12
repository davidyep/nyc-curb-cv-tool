from __future__ import annotations

import cv2
import numpy as np

from app.schemas import (
    Detection,
    DetectionInZone,
    LegalityDecision,
    ZoneDefinition,
)

# Zone fill colors (BGR)
_ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "parking": (0, 200, 0),
    "no_parking": (0, 0, 220),
    "bus_lane": (0, 165, 255),
    "bike_lane": (255, 200, 0),
    "loading_zone": (255, 180, 50),
    "fire_hydrant": (0, 0, 180),
    "double_parking": (0, 0, 255),
    "travel_lane": (180, 180, 180),
}

# Status colors for bounding boxes (BGR)
_STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "legal": (0, 200, 0),
    "likely_illegal": (0, 0, 230),
    "uncertain": (0, 220, 255),
    "in_transit": (200, 200, 200),
}

# Detection label display colors (BGR) — non-vehicle types
_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (255, 160, 50),
    "bicycle": (255, 200, 0),
}


def draw_annotations(
    image: np.ndarray,
    detections: list[Detection],
    assignments: list[DetectionInZone],
    decisions: list[LegalityDecision],
    zones: list[ZoneDefinition],
) -> np.ndarray:
    """Draw zone overlays, bounding boxes, and status labels on the image."""
    annotated = image.copy()

    # Build a decision lookup by track_id
    decision_map: dict[str, LegalityDecision] = {d.track_id: d for d in decisions}

    # 1. Draw zone polygons with semi-transparent fill
    overlay = annotated.copy()
    for zone in zones:
        if len(zone.polygon) < 3:
            continue
        pts = np.array(zone.polygon, dtype=np.int32)
        color = _ZONE_COLORS.get(zone.zone_type, (150, 150, 150))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        label = zone.label or zone.zone_type.replace("_", " ").title()
        cv2.putText(
            annotated, label, (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

    # 2. Draw detection bounding boxes
    for assignment in assignments:
        det = assignment.detection
        decision = decision_map.get(det.detection_id)
        status = decision.status if decision else "uncertain"

        # Pick color: pedestrians/cyclists get their own; vehicles by status
        if det.label in _LABEL_COLORS:
            color = _LABEL_COLORS[det.label]
        else:
            color = _STATUS_COLORS.get(status, (150, 150, 150))

        x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
        x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

        thickness = 2 if status != "in_transit" else 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        parts = [det.label]
        if det.classification != "unknown":
            parts.append(det.classification)
        if status == "in_transit":
            parts.append("moving")
        elif det.label not in ("person", "bicycle"):
            parts.append(status)
        if assignment.is_in_transit is False and assignment.zone and assignment.zone.zone_type == "double_parking":
            parts.append("DOUBLE PARKED")

        text = " | ".join(parts)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(
            annotated, (x1, y1 - text_size[1] - 6), (x1 + text_size[0] + 4, y1),
            color, -1,
        )
        cv2.putText(
            annotated, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

    return annotated

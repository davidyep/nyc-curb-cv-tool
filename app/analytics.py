from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from app.schemas import LegalityDecision, VehicleObservation

if TYPE_CHECKING:
    from app.schemas import Detection, DetectionInZone, ZoneDefinition


def occupancy_rate(observations: list[VehicleObservation], capacity: int = 20) -> float:
    occupied = sum(1 for o in observations if o.lane_type in {"parking", "travel", "bus", "bike"})
    return round(min(occupied / capacity, 1.0), 3)


def summarize_decisions(decisions: list[LegalityDecision]) -> dict[str, int]:
    counter = Counter(d.status for d in decisions)
    return {
        "legal": counter.get("legal", 0),
        "likely_illegal": counter.get("likely_illegal", 0),
        "uncertain": counter.get("uncertain", 0),
    }


# ---------------------------------------------------------------------------
# CV-specific analytics
# ---------------------------------------------------------------------------


def zone_occupancy(
    assignments: list[DetectionInZone],
    zones: list[ZoneDefinition],
) -> dict[str, float]:
    """Compute occupancy count per zone."""
    zone_counts: dict[str, int] = {z.zone_id: 0 for z in zones}
    for a in assignments:
        if a.zone and a.zone.zone_id in zone_counts:
            zone_counts[a.zone.zone_id] += 1
    return {zid: float(count) for zid, count in zone_counts.items()}


def violation_breakdown(decisions: list[LegalityDecision]) -> dict[str, int]:
    """Count occurrences of each unique reason_code across all decisions."""
    counts: dict[str, int] = {}
    for d in decisions:
        for code in d.reason_codes:
            counts[code] = counts.get(code, 0) + 1
    return counts


def detection_summary(detections: list[Detection]) -> dict[str, int]:
    """Count detections by label (car, truck, bus, motorcycle, bicycle)."""
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    return counts

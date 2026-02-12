from __future__ import annotations

from shapely.geometry import Point, Polygon

from app.schemas import (
    COCO_TO_VEHICLE_TYPE,
    ZONE_TO_LANE_TYPE,
    Detection,
    DetectionInZone,
    VehicleObservation,
    ZoneDefinition,
)


class ZoneAnalyzer:
    """Maps vehicle detections to user-defined and auto-detected zones.

    Uses Shapely point-in-polygon tests.
    """

    def __init__(self) -> None:
        self._zone_polygons: list[tuple[ZoneDefinition, Polygon]] = []

    def load_zones(self, zones: list[ZoneDefinition]) -> None:
        """Convert ZoneDefinition list to Shapely Polygon objects."""
        self._zone_polygons = [
            (z, Polygon(z.polygon)) for z in zones if len(z.polygon) >= 3
        ]

    def assign_detections_to_zones(
        self, detections: list[Detection]
    ) -> list[DetectionInZone]:
        """For each detection, find which zone its ground point falls within."""
        results: list[DetectionInZone] = []
        for det in detections:
            zone = self._find_zone(det.center_x, det.center_y)
            vehicle_type = COCO_TO_VEHICLE_TYPE.get(det.label, "other")
            lane_type = ZONE_TO_LANE_TYPE.get(zone.zone_type, "unknown") if zone else "unknown"

            results.append(
                DetectionInZone(
                    detection=det,
                    zone=zone,
                    vehicle_type=vehicle_type,
                    lane_type=lane_type,
                )
            )
        return results

    def detections_to_observations(
        self, assignments: list[DetectionInZone]
    ) -> list[VehicleObservation]:
        """Convert DetectionInZone list to VehicleObservation list.

        This is the bridge between the CV pipeline and the existing
        RulesEngine.evaluate() interface.
        """
        observations: list[VehicleObservation] = []
        for a in assignments:
            zone_type = a.zone.zone_type if a.zone else None
            observations.append(
                VehicleObservation(
                    track_id=a.detection.detection_id,
                    vehicle_type=a.vehicle_type,
                    lane_type=a.lane_type,
                    is_double_parked=(zone_type == "double_parking"),
                    is_obstructing=(
                        zone_type in ("bus_lane", "bike_lane", "fire_hydrant")
                        and a.vehicle_type not in ("bus", "bike")
                    ),
                    curb_distance_m=0.0,
                    dwell_time_seconds=0,
                )
            )
        return observations

    def _find_zone(self, x: float, y: float) -> ZoneDefinition | None:
        """Return the first zone whose polygon contains the point."""
        point = Point(x, y)
        for zone_def, shapely_poly in self._zone_polygons:
            if shapely_poly.contains(point):
                return zone_def
        return None

from __future__ import annotations

import uuid

import cv2
import numpy as np

from app.schemas import ZoneDefinition


class LaneDetector:
    """Detects bus and bike lane markings using HSV color segmentation.

    NYC bus lanes use red / terracotta paint.
    NYC bike lanes use green paint.
    Thresholds are configurable for tuning under different lighting.
    """

    def __init__(
        self,
        bus_hsv_lower: tuple[int, int, int] = (0, 80, 80),
        bus_hsv_upper: tuple[int, int, int] = (15, 255, 255),
        bike_hsv_lower: tuple[int, int, int] = (35, 80, 80),
        bike_hsv_upper: tuple[int, int, int] = (85, 255, 255),
        min_contour_area: int = 5000,
    ) -> None:
        self.bus_hsv_lower = np.array(bus_hsv_lower, dtype=np.uint8)
        self.bus_hsv_upper = np.array(bus_hsv_upper, dtype=np.uint8)
        self.bike_hsv_lower = np.array(bike_hsv_lower, dtype=np.uint8)
        self.bike_hsv_upper = np.array(bike_hsv_upper, dtype=np.uint8)
        self.min_contour_area = min_contour_area

    def detect_lanes(self, image: np.ndarray) -> list[ZoneDefinition]:
        """Detect bus/bike lanes via color segmentation.

        Returns auto-detected ZoneDefinition polygons that supplement
        user-drawn zones.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        zones: list[ZoneDefinition] = []
        zones.extend(self._find_zones(hsv, self.bus_hsv_lower, self.bus_hsv_upper, "bus_lane"))
        zones.extend(self._find_zones(hsv, self.bike_hsv_lower, self.bike_hsv_upper, "bike_lane"))
        return zones

    def _find_zones(
        self,
        hsv: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        zone_type: str,
    ) -> list[ZoneDefinition]:
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        zones: list[ZoneDefinition] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            polygon = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            zones.append(
                ZoneDefinition(
                    zone_id=f"auto_{zone_type}_{uuid.uuid4().hex[:6]}",
                    zone_type=zone_type,  # type: ignore[arg-type]
                    polygon=polygon,
                    label=f"Auto-detected {zone_type.replace('_', ' ')}",
                )
            )

        return zones

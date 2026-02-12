from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from app.schemas import BoundingBox, Detection

# COCO class IDs for vehicles we care about
_VEHICLE_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """Wraps ultralytics YOLOv8 for vehicle and cyclist detection."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run YOLOv8 inference on a BGR numpy array.

        Returns Detection objects filtered to vehicle classes only.
        """
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []

        for idx, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            if class_id not in _VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            # Bottom-center is a better ground-plane proxy than centroid
            center_x = (x1 + x2) / 2
            center_y = y2

            detections.append(
                Detection(
                    detection_id=f"det_{idx:04d}",
                    label=_VEHICLE_CLASSES[class_id],
                    confidence=round(float(box.conf[0]), 3),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    center_x=center_x,
                    center_y=center_y,
                )
            )

        return detections

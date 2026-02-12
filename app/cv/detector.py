from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from app.schemas import BoundingBox, Detection

# COCO class IDs we detect
_DETECT_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Classification heuristics based on COCO label
_COMMERCIAL_LABELS = {"truck", "bus"}
_PRIVATE_LABELS = {"car", "motorcycle"}


class VehicleDetector:
    """Wraps ultralytics YOLOv8 for vehicle, cyclist, and pedestrian detection.

    Uses yolov8m (medium) by default for better accuracy at distinguishing
    similar classes like bus vs truck.
    """

    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        confidence_threshold: float = 0.30,
    ) -> None:
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run YOLOv8 inference on a BGR numpy array.

        Returns Detection objects for vehicles, cyclists, and pedestrians.
        """
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []

        for idx, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            if class_id not in _DETECT_CLASSES:
                continue

            label = _DETECT_CLASSES[class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

            # Bottom-center is a better ground-plane proxy than centroid
            center_x = (x1 + x2) / 2
            center_y = y2

            # Classify as commercial / private / unknown
            if label in _COMMERCIAL_LABELS:
                classification = "commercial"
            elif label in _PRIVATE_LABELS:
                classification = "private"
            else:
                classification = "unknown"

            detections.append(
                Detection(
                    detection_id=f"det_{idx:04d}",
                    label=label,
                    confidence=round(float(box.conf[0]), 3),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    center_x=center_x,
                    center_y=center_y,
                    classification=classification,
                    is_stationary=True,  # single-frame: assume stationary
                )
            )

        return detections

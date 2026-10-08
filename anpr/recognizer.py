import statistics
from dataclasses import dataclass

import numpy as np
from fast_alpr import ALPR

from anpr.plates import is_valid_polish_plate, normalize


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class PlateReading:
    text: str
    confidence: float
    box: Box


CPU_ONLY = ["CPUExecutionProvider"]


class PlateRecognizer:
    def __init__(self, min_plate_width: int = 80) -> None:
        self.alpr = ALPR(detector_providers=CPU_ONLY, ocr_providers=CPU_ONLY)
        self.min_plate_width = min_plate_width

    def read(self, frame: np.ndarray) -> list[PlateReading]:
        readings = []
        for result in self.alpr.predict(frame):
            bbox = result.detection.bounding_box
            if result.ocr is None or bbox.x2 - bbox.x1 < self.min_plate_width:
                continue
            text = normalize(result.ocr.text)
            if not is_valid_polish_plate(text):
                continue
            confidence = result.ocr.confidence
            if isinstance(confidence, list):
                confidence = statistics.mean(confidence)
            box = Box(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
            readings.append(PlateReading(text, float(confidence), box))
        return readings

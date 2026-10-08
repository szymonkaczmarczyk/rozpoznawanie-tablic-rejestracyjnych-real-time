import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from anpr.recognizer import Box, PlateReading


@dataclass(frozen=True)
class Vehicle:
    plate: str
    confidence: float
    readings: int
    first_seen: float
    last_seen: float
    snapshot: np.ndarray | None


@dataclass(eq=False)
class Track:
    first_seen: float
    last_seen: float
    last_box: Box
    readings: list[PlateReading] = field(default_factory=list)
    snapshot: np.ndarray | None = None
    snapshot_confidence: float = 0.0

    def add(self, time: float, reading: PlateReading, frame: np.ndarray | None) -> None:
        self.readings.append(reading)
        self.last_seen = time
        self.last_box = reading.box
        if frame is not None and reading.confidence > self.snapshot_confidence:
            self.snapshot = frame.copy()
            self.snapshot_confidence = reading.confidence


def vote(readings: list[PlateReading]) -> tuple[str, float]:
    length_weights: Counter[int] = Counter()
    for reading in readings:
        length_weights[len(reading.text)] += reading.confidence
    length = length_weights.most_common(1)[0][0]
    same_length = [reading for reading in readings if len(reading.text) == length]

    characters = []
    for position in range(length):
        weights: Counter[str] = Counter()
        for reading in same_length:
            weights[reading.text[position]] += reading.confidence
        characters.append(weights.most_common(1)[0][0])
    plate = "".join(characters)

    total = sum(reading.confidence for reading in readings)
    agreeing = sum(reading.confidence for reading in readings if reading.text == plate)
    return plate, agreeing / total


def is_same_plate_position(previous: Box, current: Box) -> bool:
    (px, py), (cx, cy) = previous.center, current.center
    distance = math.hypot(cx - px, cy - py)
    return distance < 1.5 * max(previous.width, current.width)


class PlateTracker:
    def __init__(self, max_gap_seconds: float = 2.0, min_readings: int = 3, repeat_cooldown_seconds: float = 60.0) -> None:
        self.max_gap_seconds = max_gap_seconds
        self.min_readings = min_readings
        self.repeat_cooldown_seconds = repeat_cooldown_seconds
        self.tracks: list[Track] = []
        self.last_emitted: dict[str, float] = {}

    def update(self, time: float, readings: list[PlateReading], frame: np.ndarray | None = None) -> list[Vehicle]:
        for reading in readings:
            track = self._find_track(reading.box)
            if track is None:
                track = Track(first_seen=time, last_seen=time, last_box=reading.box)
                self.tracks.append(track)
            track.add(time, reading, frame)

        finished = [track for track in self.tracks if time - track.last_seen > self.max_gap_seconds]
        self.tracks = [track for track in self.tracks if track not in finished]
        return self._to_vehicles(finished)

    def flush(self) -> list[Vehicle]:
        finished, self.tracks = self.tracks, []
        return self._to_vehicles(finished)

    def _find_track(self, box: Box) -> Track | None:
        candidates = [track for track in self.tracks if is_same_plate_position(track.last_box, box)]
        if not candidates:
            return None
        return min(candidates, key=lambda track: math.dist(track.last_box.center, box.center))

    def _to_vehicles(self, tracks: list[Track]) -> list[Vehicle]:
        vehicles = []
        for track in sorted(tracks, key=lambda track: track.first_seen):
            if len(track.readings) < self.min_readings:
                continue
            plate, confidence = vote(track.readings)
            previous_last_seen = self.last_emitted.get(plate)
            self.last_emitted[plate] = track.last_seen
            if previous_last_seen is not None and track.first_seen - previous_last_seen < self.repeat_cooldown_seconds:
                continue
            vehicles.append(
                Vehicle(
                    plate=plate,
                    confidence=confidence,
                    readings=len(track.readings),
                    first_seen=track.first_seen,
                    last_seen=track.last_seen,
                    snapshot=track.snapshot,
                )
            )
        return vehicles

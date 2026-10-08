import sqlite3
import time as clock
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from anpr import database
from anpr.recognizer import PlateReading, PlateRecognizer
from anpr.tracker import PlateTracker, Vehicle

WINDOW_NAME = "ANPR - q zamyka"
BOX_COLOR = (0, 200, 0)


def draw_readings(frame: np.ndarray, readings: list[PlateReading]) -> np.ndarray:
    annotated = frame.copy()
    for reading in readings:
        box = reading.box
        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), BOX_COLOR, 2)
        label = f"{reading.text} {reading.confidence:.0%}"
        cv2.putText(annotated, label, (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2)
    return annotated


def save_snapshot(vehicle: Vehicle, snapshots_dir: Path, camera_id: int, seen_at: datetime) -> str | None:
    if vehicle.snapshot is None:
        return None
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    path = snapshots_dir / f"{camera_id}_{seen_at:%Y%m%d_%H%M%S}_{vehicle.plate}.jpg"
    cv2.imwrite(str(path), vehicle.snapshot)
    return str(path)


def store_vehicle(
    connection: sqlite3.Connection,
    camera_id: int,
    vehicle: Vehicle,
    started_at: datetime,
    snapshots_dir: Path,
) -> None:
    first_seen = started_at + timedelta(seconds=vehicle.first_seen)
    last_seen = started_at + timedelta(seconds=vehicle.last_seen)
    snapshot_path = save_snapshot(vehicle, snapshots_dir, camera_id, first_seen)
    database.save_passage(
        connection,
        camera_id,
        vehicle.plate,
        vehicle.confidence,
        vehicle.readings,
        first_seen,
        last_seen,
        snapshot_path,
    )
    print(f"{first_seen:%H:%M:%S}  {vehicle.plate:<9} pewność {vehicle.confidence:.0%}  odczytów {vehicle.readings}")


def process_source(
    source: str,
    connection: sqlite3.Connection,
    camera_id: int,
    recognizer: PlateRecognizer,
    every: int = 3,
    show: bool = False,
    snapshots_dir: Path = Path("data/snapshots"),
) -> list[Vehicle]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Nie udało się otworzyć źródła wideo: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    started_at = datetime.now()
    started_clock = clock.monotonic()
    tracker = PlateTracker()
    vehicles: list[Vehicle] = []
    readings: list[PlateReading] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time = frame_index / fps

            if frame_index % every == 0:
                readings = recognizer.read(frame)
                for vehicle in tracker.update(time, readings, frame):
                    store_vehicle(connection, camera_id, vehicle, started_at, snapshots_dir)
                    vehicles.append(vehicle)

            if show:
                cv2.imshow(WINDOW_NAME, draw_readings(frame, readings))
                delay_ms = int((time - (clock.monotonic() - started_clock)) * 1000)
                if cv2.waitKey(max(1, delay_ms)) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    for vehicle in tracker.flush():
        store_vehicle(connection, camera_id, vehicle, started_at, snapshots_dir)
        vehicles.append(vehicle)
    return vehicles

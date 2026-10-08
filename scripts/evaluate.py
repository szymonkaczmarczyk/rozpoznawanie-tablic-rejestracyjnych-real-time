import csv
import logging
import time
from pathlib import Path

import onnxruntime

from anpr import database
from anpr.pipeline import process_source
from anpr.recognizer import PlateRecognizer

VIDEOS_DIR = Path("data/videos")
GROUND_TRUTH = Path("data/ground_truth.csv")
EVALUATION_DB = Path("data/evaluation.db")


def load_ground_truth() -> dict[str, str]:
    with GROUND_TRUTH.open(encoding="utf-8") as file:
        return {row["SRC"]: row["Vehicle_registration_plate"] for row in csv.DictReader(file, delimiter=";")}


def main() -> None:
    logging.disable(logging.WARNING)
    onnxruntime.set_default_logger_severity(3)
    EVALUATION_DB.unlink(missing_ok=True)
    connection = database.connect(EVALUATION_DB)
    recognizer = PlateRecognizer()
    expected_plates = load_ground_truth()

    correct = 0
    started = time.perf_counter()
    for name, expected in expected_plates.items():
        video = VIDEOS_DIR / f"agh_{name}.avi"
        camera_id = database.get_or_create_camera(connection, f"AGH {name}", "wjazd", str(video))
        vehicles = process_source(str(video), connection, camera_id, recognizer, snapshots_dir=Path("data/snapshots/evaluation"))
        plates = [vehicle.plate for vehicle in vehicles]
        is_correct = plates == [expected]
        correct += is_correct
        print(f"{'OK ' if is_correct else 'BŁĄD'} {name:<6} oczekiwano {expected:<9} odczytano {', '.join(plates) or '-'}")

    elapsed = time.perf_counter() - started
    print(f"\nSkuteczność: {correct}/{len(expected_plates)} ({correct / len(expected_plates):.0%}), czas {elapsed:.0f} s")


if __name__ == "__main__":
    main()

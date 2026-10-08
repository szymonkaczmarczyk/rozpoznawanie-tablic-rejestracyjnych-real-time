import argparse
import logging

import onnxruntime

from anpr import database
from anpr.pipeline import process_source
from anpr.plates import mask
from anpr.recognizer import PlateRecognizer

DEFAULT_DB = "data/anpr.db"


def run(args: argparse.Namespace) -> None:
    connection = database.connect(args.db)
    camera_id = database.get_or_create_camera(connection, args.camera, args.direction, args.source)
    recognizer = PlateRecognizer()
    print(f"Kamera: {args.camera} ({args.direction}), źródło: {args.source}")
    vehicles = process_source(args.source, connection, camera_id, recognizer, every=args.every, show=args.show)
    print(f"Zapisano przejazdów: {len(vehicles)}")


def report(args: argparse.Namespace) -> None:
    connection = database.connect(args.db)
    show_plate = (lambda plate: plate) if args.full else mask

    print("Ostatnie przejazdy")
    for row in database.latest_passages(connection, args.limit):
        print(f"  {row['first_seen']}  {show_plate(row['plate']):<9} {row['confidence']:.0%}  {row['camera']} ({row['direction']})")

    print("\nWjazdy na godzinę")
    for row in database.entries_per_hour(connection):
        print(f"  {row['hour']}  {row['entries']}")

    print("\nWjazdy według województwa")
    for row in database.entries_per_voivodeship(connection):
        print(f"  {row['voivodeship']:<20} {row['entries']}")


def main() -> None:
    logging.disable(logging.WARNING)
    onnxruntime.set_default_logger_severity(3)
    parser = argparse.ArgumentParser(prog="anpr", description="Rozpoznawanie tablic rejestracyjnych z zapisem do SQLite")
    parser.add_argument("--db", default=DEFAULT_DB, help="ścieżka do pliku bazy SQLite")
    commands = parser.add_subparsers(required=True)

    run_parser = commands.add_parser("run", help="analizuj film lub strumień z kamery")
    run_parser.add_argument("source", help="plik wideo albo adres strumienia (rtsp://, http://)")
    run_parser.add_argument("--camera", required=True, help="nazwa kamery zapisywana w bazie")
    run_parser.add_argument("--direction", choices=["wjazd", "wyjazd"], default="wjazd")
    run_parser.add_argument("--every", type=int, default=3, help="analizuj co N-tą klatkę")
    run_parser.add_argument("--show", action="store_true", help="pokaż podgląd w oknie")
    run_parser.set_defaults(handler=run)

    report_parser = commands.add_parser("report", help="pokaż statystyki z bazy")
    report_parser.add_argument("--limit", type=int, default=20)
    report_parser.add_argument("--full", action="store_true", help="pokaż pełne numery zamiast zamaskowanych")
    report_parser.set_defaults(handler=report)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

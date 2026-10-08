import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return connection


def get_or_create_camera(connection: sqlite3.Connection, name: str, direction: str, source: str) -> int:
    row = connection.execute("SELECT id FROM cameras WHERE name = ?", (name,)).fetchone()
    if row is not None:
        return row["id"]
    cursor = connection.execute(
        "INSERT INTO cameras (name, direction, source) VALUES (?, ?, ?)",
        (name, direction, source),
    )
    connection.commit()
    return cursor.lastrowid


def save_passage(
    connection: sqlite3.Connection,
    camera_id: int,
    plate: str,
    confidence: float,
    readings: int,
    first_seen: datetime,
    last_seen: datetime,
    snapshot_path: str | None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO passages (camera_id, plate, confidence, readings, first_seen, last_seen, snapshot_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            camera_id,
            plate,
            confidence,
            readings,
            first_seen.isoformat(timespec="seconds"),
            last_seen.isoformat(timespec="seconds"),
            snapshot_path,
        ),
    )
    connection.commit()
    return cursor.lastrowid


def latest_passages(connection: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT passages.first_seen, passages.plate, passages.confidence, passages.readings,
               cameras.name AS camera, cameras.direction
        FROM passages
        JOIN cameras ON cameras.id = passages.camera_id
        ORDER BY passages.first_seen DESC, passages.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def entries_per_hour(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT strftime('%Y-%m-%d %H:00', passages.first_seen) AS hour, COUNT(*) AS entries
        FROM passages
        JOIN cameras ON cameras.id = passages.camera_id
        WHERE cameras.direction = 'wjazd'
        GROUP BY hour
        ORDER BY hour
        """
    ).fetchall()


def entries_per_voivodeship(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT COALESCE(voivodeships.name, 'inne') AS voivodeship, COUNT(*) AS entries
        FROM passages
        JOIN cameras ON cameras.id = passages.camera_id
        LEFT JOIN voivodeships ON voivodeships.letter = substr(passages.plate, 1, 1)
        WHERE cameras.direction = 'wjazd'
        GROUP BY voivodeship
        ORDER BY entries DESC
        """
    ).fetchall()

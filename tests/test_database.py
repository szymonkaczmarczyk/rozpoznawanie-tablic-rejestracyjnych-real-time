from datetime import datetime

from anpr import database


def test_passages_are_saved_and_counted():
    connection = database.connect(":memory:")
    entry = database.get_or_create_camera(connection, "Brama", "wjazd", "film.avi")
    exit_ = database.get_or_create_camera(connection, "Wyjazd", "wyjazd", "film2.avi")
    assert database.get_or_create_camera(connection, "Brama", "wjazd", "film.avi") == entry

    seen = datetime(2026, 10, 8, 8, 15)
    for plate in ["KR123AB", "KR456CD", "WA12345"]:
        database.save_passage(connection, entry, plate, 0.9, 10, seen, seen, None)
    database.save_passage(connection, exit_, "XX12345", 0.9, 10, seen, seen, None)

    assert len(database.latest_passages(connection)) == 4
    assert [tuple(row) for row in database.entries_per_hour(connection)] == [("2026-10-08 08:00", 3)]
    assert [tuple(row) for row in database.entries_per_voivodeship(connection)] == [
        ("małopolskie", 2),
        ("mazowieckie", 1),
    ]

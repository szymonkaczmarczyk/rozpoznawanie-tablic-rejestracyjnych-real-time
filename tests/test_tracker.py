from anpr.recognizer import Box, PlateReading
from anpr.tracker import PlateTracker, vote

BOX = Box(100, 100, 200, 130)


def reading(text: str, confidence: float = 0.9, box: Box = BOX) -> PlateReading:
    return PlateReading(text, confidence, box)


def test_vote_fixes_single_wrong_character():
    plate, confidence = vote([reading("KR123AB"), reading("KR123AB"), reading("KR12ZAB")])
    assert plate == "KR123AB"
    assert round(confidence, 2) == 0.67


def test_vote_ignores_truncated_readings():
    plate, _ = vote([reading("KRA1BC2"), reading("KRA1BC2"), reading("KRA1BC")])
    assert plate == "KRA1BC2"


def test_one_vehicle_is_emitted_after_it_disappears():
    tracker = PlateTracker(max_gap_seconds=2.0, min_readings=3)
    for second in range(5):
        assert tracker.update(second, [reading("KR123AB")]) == []
    vehicles = tracker.update(10.0, [])
    assert [vehicle.plate for vehicle in vehicles] == ["KR123AB"]
    assert vehicles[0].readings == 5


def test_short_flicker_is_ignored():
    tracker = PlateTracker(min_readings=3)
    tracker.update(0.0, [reading("KR123AB")])
    assert tracker.flush() == []


def test_two_plates_far_apart_are_two_vehicles():
    tracker = PlateTracker(min_readings=1)
    far_box = Box(900, 500, 1000, 530)
    tracker.update(0.0, [reading("KR123AB"), reading("WA12345", box=far_box)])
    assert sorted(vehicle.plate for vehicle in tracker.flush()) == ["KR123AB", "WA12345"]


def test_same_plate_soon_after_is_not_counted_twice():
    tracker = PlateTracker(min_readings=1, repeat_cooldown_seconds=60.0)
    tracker.update(0.0, [reading("KR123AB")])
    assert len(tracker.update(5.0, [])) == 1
    tracker.update(10.0, [reading("KR123AB")])
    assert tracker.update(15.0, []) == []

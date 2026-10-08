from anpr.plates import is_valid_polish_plate, mask, normalize


def test_normalize_removes_spaces_and_symbols():
    assert normalize("kr 123-ab") == "KR123AB"


def test_accepts_polish_plates():
    for plate in ["KR123AB", "KLI12345", "W12345X", "KRA1BC2"]:
        assert is_valid_polish_plate(plate)


def test_rejects_garbage():
    for text in ["", "KR1", "123456", "KR1234567"]:
        assert not is_valid_polish_plate(text)


def test_mask_keeps_district_prefix():
    assert mask("KR123AB") == "KR1****"

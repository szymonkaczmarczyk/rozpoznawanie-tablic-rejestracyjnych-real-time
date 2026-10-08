import re

POLISH_PLATE = re.compile(r"^[A-Z]{1,3}[A-Z0-9]{3,6}$")


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_valid_polish_plate(plate: str) -> bool:
    return 4 <= len(plate) <= 8 and POLISH_PLATE.match(plate) is not None


def mask(plate: str) -> str:
    visible = 3
    return plate[:visible] + "*" * (len(plate) - visible)

# Rozpoznawanie tablic rejestracyjnych

**Polski** · [English](README.en.md)

> Wideo wchodzi, przejazdy wychodzą.

System ANPR (automatyczne rozpoznawanie tablic rejestracyjnych) w Pythonie. Program czyta wideo z pliku albo
z kamery w sieci, znajduje na klatkach polskie tablice, odczytuje numer, łączy odczyty jednego auta w jeden
przejazd i zapisuje go do bazy SQLite razem ze zdjęciem. Z bazy od razu da się wyciągnąć proste statystyki:
ostatnie przejazdy, wjazdy na godzinę i wjazdy według województwa.

Projekt powstał też jako nauka Pythona i SQL-a, dlatego baza to czysty SQL przez `sqlite3`, bez ORM-a, a cały
kod jest objaśniony linia po linii w [`docs/jak-dziala-kod.md`](docs/jak-dziala-kod.md).

**Wynik:** 22 z 22 poprawnie odczytanych tablic na nagraniach testowych z parkingu AGH, przy przetwarzaniu
ok. 6 razy szybszym niż czas rzeczywisty na Macu z Apple M4 (CPU).

## Dla kogo

- **Dla osób uczących się computer vision**, które chcą zobaczyć cały system od klatki wideo do rekordu
  w bazie, a nie tylko pojedynczy model.
- **Dla właściciela parkingu, bramy albo osiedla**, który chce sprawdzić, czy jego kamera nadaje się do
  odczytu tablic, zanim kupi gotowy system.
- **Dla rekruterów i deweloperów** - mały, czytelny kod (ok. 450 linii) z testami i dokumentacją.

Czym projekt **nie jest**: nie jest gotowym systemem do kontroli dostępu. Nie steruje szlabanem, nie ma panelu
www ani listy dozwolonych numerów.

## Jak to działa

```
plik wideo albo strumień (rtsp://, http://)
   │
   ▼
pipeline.py      czyta klatki, co 3. klatkę oddaje do analizy
   │
   ▼
recognizer.py    YOLO znajduje tablicę, OCR czyta napis, plates.py odrzuca śmieci
   │
   ▼
tracker.py       łączy odczyty w ślad jednego auta i głosuje nad numerem
   │
   ▼
database.py      zapisuje przejazd do SQLite, zdjęcie do data/snapshots
```

1. **Detekcja i OCR** - biblioteka [fast-alpr](https://github.com/ankandrew/fast-alpr) z modelami ONNX:
   detektor `yolo-v9-t-384-license-plate-end2end` i OCR `cct-xs-v2-global-model`. Działa na ONNX Runtime
   na CPU, ok. 13 ms na klatkę.
2. **Filtrowanie** - tablice węższe niż 80 px są pomijane, bo OCR gubi wtedy znaki. Numer musi pasować do
   polskiego wzorca: 4-8 znaków, 1-3 litery na początku.
3. **Śledzenie** - odczyt trafia do istniejącego śladu, jeśli środek tablicy jest bliżej niż 1,5 jej
   szerokości. Ślad kończy się po 2 s bez odczytu i musi mieć co najmniej 3 odczyty.
4. **Głosowanie** - najpierw wygrywa najczęstsza długość numeru, potem znak po znaku większość ważona
   pewnością OCR. Pewność przejazdu to udział odczytów zgodnych z wynikiem.
5. **Zapis** - ten sam numer na tej samej kamerze w ciągu 60 s nie jest liczony drugi raz.

## Funkcje

- Wideo z pliku albo strumienia, który otworzy OpenCV (`rtsp://`, `http://.../stream.m3u8`).
- Podgląd na żywo z ramkami i odczytem (`--show`), w tempie nagrania.
- Baza SQLite: kamery, przejazdy, województwa po pierwszej literze tablicy.
- Raport w terminalu z numerami domyślnie zamaskowanymi (`KR1****`).
- Skrypt oceny skuteczności na nagraniach z plikiem wzorcowych numerów.
- 11 testów jednostkowych (pytest) na zmyślonych numerach.

Przykładowy raport (dane zmyślone):

```
Ostatnie przejazdy
  2026-10-08T20:02:53  KR1****   97%  Brama AGH (wjazd)
  2026-10-08T20:02:34  SK2****   94%  Brama AGH (wjazd)

Wjazdy na godzinę
  2026-10-08 19:00  6
  2026-10-08 20:00  4

Wjazdy według województwa
  małopolskie          8
  śląskie              2
```

## Uruchomienie

Wymagany Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Analiza filmu i raport:

```bash
.venv/bin/python -m anpr run data/videos/film.avi --camera "Brama" --show
.venv/bin/python -m anpr report
```

| Polecenie | Co robi |
|---|---|
| `anpr run ŹRÓDŁO --camera NAZWA` | Analizuje film albo strumień i zapisuje przejazdy |
| `--direction wjazd\|wyjazd` | Kierunek kamery zapisywany w bazie (domyślnie `wjazd`) |
| `--every N` | Analizuje co N-tą klatkę (domyślnie 3) |
| `--show` | Pokazuje okno podglądu, `q` zamyka |
| `anpr report` | Ostatnie przejazdy i statystyki |
| `anpr report --full` | To samo, ale z pełnymi numerami |
| `--db ŚCIEŻKA` | Inny plik bazy (domyślnie `data/anpr.db`) |
| `python -m scripts.evaluate` | Skuteczność na nagraniach z `data/videos` i `data/ground_truth.csv` |

Modele ONNX pobierają się automatycznie przy pierwszym uruchomieniu.

## Dane testowe

Skuteczność była mierzona na nagraniach z bazy
[AGH Parking Database](https://qoe.agh.edu.pl/parking-database/): 22 filmy 1280x720, 25 FPS, po jednym aucie
w każdym. Nagrań nie ma w repozytorium. Przy publikacji wyników trzeba cytować stronę bazy i pracę Leszczuk
i in. (2011).

Kamera, żeby odczyt był sensowny, powinna dawać tablicę o szerokości co najmniej 80-100 px i patrzeć pod
kątem mniejszym niż 30° do kierunku jazdy. Większość publicznych kamer panoramicznych ma na to za małą
rozdzielczość.

## Struktura

```
anpr/
  plates.py       normalizacja, walidacja i maskowanie numeru
  recognizer.py   detekcja i OCR jednej klatki
  tracker.py      śledzenie tablic między klatkami i głosowanie
  database.py     zapis przejazdów i zapytania SQL
  schema.sql      tabele: kamery, przejazdy, województwa
  pipeline.py     pętla wideo, zapis przejazdów i zdjęć
  __main__.py     wiersz poleceń (run, report)
scripts/
  evaluate.py     pomiar skuteczności na nagraniach testowych
tests/            testy jednostkowe
docs/
  jak-dziala-kod.md   objaśnienie kodu linia po linii
data/             nagrania, bazy i zdjęcia (poza gitem)
```

## Prywatność

Numer tablicy razem z miejscem i czasem przejazdu może być daną osobową (RODO). Dlatego:

- bazy, nagrania i zdjęcia aut są w `.gitignore` i nigdy nie trafiają do repozytorium,
- raport domyślnie maskuje numery,
- testy używają wyłącznie zmyślonych numerów.

Używaj programu tylko na nagraniach, do których masz prawo, i z krótkim czasem przechowywania danych.

## Plany

- Wirtualna linia na obrazie zamiast kierunku z konfiguracji kamery.
- Transmisje na żywo (RTSP, HLS, YouTube Live) w dłuższych testach.
- Panel w Streamlit: lista przejazdów, wykres wjazdów, wyszukiwarka.
- Dotrenowanie detektora na polskich tablicach.

## Dokumentacja

- [`docs/jak-dziala-kod.md`](docs/jak-dziala-kod.md) - objaśnienie każdego pliku linia po linii, droga klatki
  przez program i słowniczek pojęć Pythona.

## Licencja

[MIT](LICENSE) © 2026 Szymon Kaczmarczyk

Biblioteka fast-alpr i jej modele są na licencji MIT. Nagrania AGH mają własne warunki użycia.

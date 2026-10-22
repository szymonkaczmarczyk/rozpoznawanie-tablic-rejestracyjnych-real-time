# Jak działa kod

Objaśnienie projektu linia po linii. Numery linii odpowiadają stanowi kodu z 8 października 2026. Jeśli kod się zmieni, numery mogą się przesunąć.

Najlepiej czytać z otwartym plikiem obok.

## Spis treści

1. [Droga jednej klatki przez program](#droga-jednej-klatki-przez-program)
2. [anpr/plates.py](#anprplatespy) - numer tablicy jako tekst
3. [anpr/recognizer.py](#anprrecognizerpy) - wykrycie i odczyt tablicy na jednej klatce
4. [anpr/tracker.py](#anprtrackerpy) - łączenie odczytów w jeden samochód
5. [anpr/schema.sql](#anprschemasql) - budowa bazy danych
6. [anpr/database.py](#anprdatabasepy) - zapis i zapytania SQL
7. [anpr/pipeline.py](#anprpipelinepy) - pętla wideo
8. [anpr/\_\_main\_\_.py](#anpr__main__py) - polecenia `run` i `report`
9. [anpr/\_\_init\_\_.py](#anpr__init__py)
10. [scripts/evaluate.py](#scriptsevaluatepy) - pomiar skuteczności
11. [tests/](#tests) - testy automatyczne
12. [requirements.txt](#requirementstxt)
13. [.gitignore](#gitignore)
14. [Słowniczek pojęć Pythona](#słowniczek-pojęć-pythona)

## Droga jednej klatki przez program

```
plik .avi albo kamera
   │
   ▼
pipeline.py      czyta klatkę po klatce, co 3. klatkę oddaje do analizy
   │
   ▼
recognizer.py    znajduje tablicę, czyta napis, odrzuca śmieci (plates.py)
   │
   ▼
tracker.py       dokłada odczyt do śladu samochodu, a gdy auto zniknie, głosuje nad numerem
   │
   ▼
database.py      zapisuje przejazd do SQLite (tabele z schema.sql)
   │
   ▼
__main__.py      polecenie report czyta bazę i drukuje statystyki
```

Każdy plik robi jedną rzecz. Dzięki temu da się go zrozumieć i przetestować osobno.

---

## anpr/plates.py

Najprostszy plik. Operacje na numerze tablicy jako zwykłym tekście, bez obrazu i bez bazy.

```python
1  import re
```

`re` to wbudowany moduł Pythona do **wyrażeń regularnych**, czyli wzorców opisujących, jak ma wyglądać tekst. Nie trzeba go instalować.

```python
3  POLISH_PLATE = re.compile(r"^[A-Z]{1,3}[A-Z0-9]{3,6}$")
```

Wzorzec polskiej tablicy:

| Fragment | Znaczenie |
|---|---|
| `r"..."` | surowy napis (raw string). Odwrotny ukośnik nie jest traktowany specjalnie. Przy wzorcach to dobry nawyk |
| `^` | początek tekstu |
| `[A-Z]{1,3}` | od 1 do 3 wielkich liter (wyróżnik powiatu, np. `KR`, `KLI`, `W`) |
| `[A-Z0-9]{3,6}` | od 3 do 6 liter lub cyfr (wyróżnik pojazdu) |
| `$` | koniec tekstu, czyli nic więcej nie może być po |

`re.compile` zamienia wzorzec w gotowy obiekt raz, przy starcie programu. Potem sprawdzanie tysięcy numerów jest szybsze. Nazwa WIELKIMI LITERAMI to w Pythonie umowny znak **stałej**, czyli wartości, której nikt nie powinien zmieniać.

```python
6  def normalize(text: str) -> str:
7      return re.sub(r"[^A-Z0-9]", "", text.upper())
```

- `def` definiuje funkcję. `text: str` mówi, że funkcja przyjmuje tekst, a `-> str`, że zwraca tekst. To **adnotacje typów**: Python ich nie wymusza, ale ułatwiają czytanie i edytor podpowiada na ich podstawie.
- `text.upper()` zamienia na wielkie litery: `"kr 123-ab"` → `"KR 123-AB"`.
- `re.sub(wzorzec, zamiennik, tekst)` zastępuje wszystko, co pasuje do wzorca. `[^A-Z0-9]` oznacza „każdy znak, który NIE jest literą A-Z ani cyfrą” (`^` wewnątrz nawiasów kwadratowych to zaprzeczenie). Zamieniamy to na pusty tekst `""`, czyli usuwamy spacje, myślniki i kropki: `"KR123AB"`.

```python
10 def is_valid_polish_plate(plate: str) -> bool:
11     return 4 <= len(plate) <= 8 and POLISH_PLATE.match(plate) is not None
```

- `bool` to typ prawda/fałsz (`True` / `False`).
- `4 <= len(plate) <= 8` to **porównanie łańcuchowe**, typowe dla Pythona. Oznacza „długość między 4 a 8 włącznie”. W innych językach trzeba by pisać `len >= 4 and len <= 8`.
- `POLISH_PLATE.match(plate)` zwraca obiekt dopasowania albo `None`, gdy tekst nie pasuje. `is not None` zamienia to na `True`/`False`.
- `and` oznacza, że oba warunki muszą być spełnione.

Po co sprawdzać długość osobno, skoro wzorzec i tak ją ogranicza (1-3 + 3-6 znaków daje 4-9)? Bo polska tablica ma najwyżej 8 znaków, a sam wzorzec przepuściłby 9.

```python
14 def mask(plate: str) -> str:
15     visible = 3
16     return plate[:visible] + "*" * (len(plate) - visible)
```

Maskowanie numeru w raportach (RODO): `KR123AB` → `KR1****`.

- `plate[:3]` to **wycinek** (slice): znaki od początku do trzeciego, bez niego, czyli indeksy 0, 1, 2.
- `"*" * 4` w Pythonie daje `"****"`. Mnożenie tekstu przez liczbę powtarza go.
- `+` skleja dwa teksty.

---

## anpr/recognizer.py

Bierze jedną klatkę wideo i zwraca listę przeczytanych tablic. Tu pracują sieci neuronowe, ale cała „magia” siedzi w bibliotece `fast-alpr`. Nasz kod ją opakowuje i filtruje wyniki.

```python
1  import statistics
2  from dataclasses import dataclass
3
4  import numpy as np
5  from fast_alpr import ALPR
6
7  from anpr.plates import is_valid_polish_plate, normalize
```

Importy w trzech grupach, oddzielonych pustą linią (konwencja PEP 8):

1. biblioteka standardowa Pythona (`statistics`, `dataclasses`)
2. biblioteki zewnętrzne z `pip install` (`numpy`, `fast_alpr`)
3. nasz własny kod (`anpr.plates`)

- `import numpy as np` importuje pod krótszą nazwą `np`. To powszechna konwencja.
- `from X import Y` bierze z modułu tylko wskazane rzeczy. Potem piszemy `normalize(...)` zamiast `anpr.plates.normalize(...)`.

```python
10 @dataclass(frozen=True)
11 class Box:
12     x1: int
13     y1: int
14     x2: int
15     y2: int
```

Prostokąt wokół tablicy na obrazie. `(x1, y1)` to lewy górny róg, `(x2, y2)` prawy dolny, w pikselach. Na obrazie oś Y rośnie w dół.

- `class` definiuje własny typ danych.
- `@dataclass` to **dekorator**. Automatycznie dopisuje do klasy konstruktor (`Box(10, 20, 110, 50)`), czytelne wypisywanie (`Box(x1=10, ...)`) i porównywanie `==`. Bez niego trzeba by te metody napisać ręcznie.
- `frozen=True` robi obiekt **niezmiennym**. Po utworzeniu nie da się zmienić `box.x1 = 5`, Python zgłosi błąd. Prostokąt z konkretnej klatki nie powinien się zmieniać, więc to chroni przed przypadkowymi błędami.

```python
17     @property
18     def width(self) -> int:
19         return self.x2 - self.x1
```

- `self` to „ten konkretny obiekt”. Każda metoda dostaje go jako pierwszy argument.
- `@property` sprawia, że metodę czyta się jak pole, bez nawiasów: `box.width`, a nie `box.width()`. Szerokość nie jest zapisana, tylko liczona przy każdym odczycie, więc nigdy nie rozjedzie się z `x1` i `x2`.

```python
21     @property
22     def center(self) -> tuple[float, float]:
23         return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
```

Środek prostokąta. `tuple[float, float]` to **krotka** (tuple) dwóch liczb, np. `(150.0, 115.0)`. Przecinek w `return a, b` sam tworzy krotkę. `/` w Pythonie 3 zawsze daje liczbę zmiennoprzecinkową (`float`), nawet z liczb całkowitych.

```python
26 @dataclass(frozen=True)
27 class PlateReading:
28     text: str
29     confidence: float
30     box: Box
```

Jeden odczyt tablicy z jednej klatki: napis, pewność OCR (0.0-1.0) i położenie. Pole `box` ma typ `Box`, więc obiekty można zagnieżdżać.

```python
33 CPU_ONLY = ["CPUExecutionProvider"]
```

Lista „dostawców obliczeń” dla ONNX Runtime (silnika uruchamiającego sieci neuronowe). Wybieramy sam procesor. CoreML (akcelerator Apple) był 2 razy szybszy, ale zgłaszał błędy na klatkach bez tablicy, a CPU i tak ma duży zapas mocy.

```python
36 class PlateRecognizer:
37     def __init__(self, min_plate_width: int = 80) -> None:
38         self.alpr = ALPR(detector_providers=CPU_ONLY, ocr_providers=CPU_ONLY)
39         self.min_plate_width = min_plate_width
```

- `__init__` to **konstruktor**, uruchamiany przy tworzeniu obiektu: `PlateRecognizer()`.
- `min_plate_width: int = 80` to argument z **wartością domyślną**. Można napisać `PlateRecognizer()` (wtedy 80) albo `PlateRecognizer(min_plate_width=100)`.
- `-> None` znaczy, że funkcja nic nie zwraca.
- `ALPR(...)` ładuje dwa modele: detektor (gdzie jest tablica) i OCR (co jest napisane). Przy pierwszym uruchomieniu pobiera je z internetu (ok. 11 MB), potem bierze z dysku. To trwa ok. sekundy, dlatego robimy to raz w konstruktorze, a nie przy każdej klatce.
- `self.alpr = ...` zapisuje model w obiekcie, żeby metoda `read` mogła go potem użyć.

```python
41     def read(self, frame: np.ndarray) -> list[PlateReading]:
42         readings = []
43         for result in self.alpr.predict(frame):
```

- `frame: np.ndarray` to klatka obrazu jako tablica NumPy o kształcie `(720, 1280, 3)`: 720 wierszy pikseli, 1280 kolumn i 3 kolory. Uwaga: OpenCV trzyma kolory w kolejności **BGR** (niebieski, zielony, czerwony), a nie RGB.
- `readings = []` tworzy pustą listę na wyniki.
- `self.alpr.predict(frame)` uruchamia obie sieci i zwraca listę wyników, jeden na każdą znalezioną tablicę.
- `for ... in ...:` to pętla po każdym elemencie listy.

```python
44             bbox = result.detection.bounding_box
45             if result.ocr is None or bbox.x2 - bbox.x1 < self.min_plate_width:
46                 continue
```

- `bbox` to prostokąt z detektora (typ z biblioteki, nie nasz `Box`).
- Odrzucamy wynik, jeśli OCR nic nie zwrócił albo tablica jest węższa niż 80 px. To poprawka z testów: przy małych tablicach OCR gubił znaki (w filmie src26 z 7 znaków robiło się 6).
- `continue` przerywa ten obrót pętli i przechodzi do następnego wyniku.
- `or` wystarczy, że jeden warunek jest prawdziwy. Python sprawdza od lewej i zatrzymuje się na pierwszym prawdziwym. Dlatego gdy `ocr` jest `None`, druga część w ogóle się nie wykona.

```python
47             text = normalize(result.ocr.text)
48             if not is_valid_polish_plate(text):
49                 continue
```

Czyścimy napis i odrzucamy wszystko, co nie wygląda jak polska tablica, np. fragment napisu z ciężarówki.

```python
50             confidence = result.ocr.confidence
51             if isinstance(confidence, list):
52                 confidence = statistics.mean(confidence)
```

Model OCR może zwrócić pewność jako jedną liczbę albo jako listę (osobno dla każdego znaku). `isinstance(x, list)` sprawdza typ. Z listy liczymy średnią, żeby dalej zawsze mieć jedną liczbę.

```python
53             box = Box(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
54             readings.append(PlateReading(text, float(confidence), box))
55         return readings
```

- Przepisujemy prostokąt na nasz typ `Box`. Reszta programu nie musi wtedy znać typów z `fast-alpr`, więc bibliotekę łatwo podmienić.
- `float(confidence)` zamienia liczbę z NumPy (`numpy.float32`) na zwykłą liczbę Pythona. Bez tego moduł `sqlite3` nie umiałby jej zapisać do bazy.
- `.append(...)` dokłada element na koniec listy.
- `return readings` jest na poziomie wcięcia `for`, czyli wykonuje się dopiero po przejściu całej pętli. **W Pythonie wcięcia decydują o strukturze kodu**.

---

## anpr/tracker.py

Najważniejsza logika projektu. Detektor widzi to samo auto w ok. 150 klatkach. Bez tego pliku w bazie byłoby 150 wpisów zamiast jednego, a pojedyncze błędy OCR trafiałyby do bazy.

Pojęcia:

- **odczyt** (`PlateReading`) to wynik z jednej klatki
- **ślad** (`Track`) to wszystkie odczyty tego samego auta, zbierane, dopóki auto jest widoczne
- **pojazd** (`Vehicle`) to gotowy wynik po zniknięciu auta: jeden numer, który idzie do bazy

```python
1  import math
2  from collections import Counter
3  from dataclasses import dataclass, field
4
5  import numpy as np
6
7  from anpr.recognizer import Box, PlateReading
```

`Counter` to słownik do liczenia: `Counter()["A"] += 1`. Nie trzeba sprawdzać, czy klucz już istnieje, bo brakujący ma wartość 0.

```python
10 @dataclass(frozen=True)
11 class Vehicle:
12     plate: str
13     confidence: float
14     readings: int
15     first_seen: float
16     last_seen: float
17     snapshot: np.ndarray | None
```

Gotowy przejazd. `first_seen` i `last_seen` to sekundy od początku nagrania (np. `0.0` i `20.4`). Na prawdziwą datę zamienia je dopiero `pipeline.py`. `np.ndarray | None` znaczy „obraz albo nic”.

```python
20 @dataclass(eq=False)
21 class Track:
22     first_seen: float
23     last_seen: float
24     last_box: Box
25     readings: list[PlateReading] = field(default_factory=list)
26     snapshot: np.ndarray | None = None
27     snapshot_confidence: float = 0.0
```

Ślad jednego auta w trakcie zbierania. Nie jest `frozen`, bo ciągle się zmienia.

- `eq=False` sprawia, że dwa ślady są „równe” tylko wtedy, gdy to dokładnie ten sam obiekt. Domyślnie dataclass porównuje wszystkie pola po kolei, a porównanie obrazów NumPy przez `==` nie daje jednej odpowiedzi tak/nie, tylko tablicę, i kończy się błędem.
- `field(default_factory=list)`: każdy nowy ślad dostaje **własną**, nową pustą listę. Gdyby napisać `= []`, wszystkie ślady dzieliłyby jedną listę. To klasyczna pułapka Pythona, dlatego dataclass na to nie pozwala.
- Pola z wartościami domyślnymi muszą być po polach bez nich.

```python
29     def add(self, time: float, reading: PlateReading, frame: np.ndarray | None) -> None:
30         self.readings.append(reading)
31         self.last_seen = time
32         self.last_box = reading.box
```

Dokłada odczyt do śladu i zapamiętuje, kiedy i gdzie auto było widziane ostatnio. `last_box` jest potrzebny, żeby w następnej klatce rozpoznać, że tablica w pobliżu to to samo auto.

```python
33         if frame is not None and reading.confidence > self.snapshot_confidence:
34             self.snapshot = frame.copy()
35             self.snapshot_confidence = reading.confidence
```

Zdjęcie do bazy: trzymamy tylko klatkę z najpewniejszym odczytem. `frame.copy()` robi osobną kopię obrazu w pamięci, żeby późniejsze operacje na klatce nie zmieniły zachowanego zdjęcia. Jedna klatka ma ok. 2,7 MB, więc nie trzymamy wszystkich.

```python
38 def vote(readings: list[PlateReading]) -> tuple[str, float]:
39     length_weights: Counter[int] = Counter()
40     for reading in readings:
41         length_weights[len(reading.text)] += reading.confidence
42     length = length_weights.most_common(1)[0][0]
```

**Głosowanie**, krok 1: jaka długość numeru wygrywa? Każdy odczyt „głosuje” na swoją długość z wagą równą pewności. Przykład dla `KRA1BC2` (0.9), `KRA1BC2` (0.9), `KRA1BC` (0.9) daje `{7: 1.8, 6: 0.9}`.

`most_common(1)` zwraca listę z jednym zwycięzcą: `[(7, 1.8)]`. `[0]` bierze pierwszy element `(7, 1.8)`, a drugie `[0]` sam klucz `7`.

```python
43     same_length = [reading for reading in readings if len(reading.text) == length]
```

**Wyrażenie listowe** (list comprehension), czyli krótki zapis pętli tworzącej listę. Czytamy: „weź każdy `reading` z `readings`, jeśli jego długość równa się `length`”. Odpowiada to:

```python
same_length = []
for reading in readings:
    if len(reading.text) == length:
        same_length.append(reading)
```

```python
45     characters = []
46     for position in range(length):
47         weights: Counter[str] = Counter()
48         for reading in same_length:
49             weights[reading.text[position]] += reading.confidence
50         characters.append(weights.most_common(1)[0][0])
51     plate = "".join(characters)
```

Krok 2: głosowanie **znak po znaku**. `range(7)` daje 0, 1, 2, ..., 6. Dla każdej pozycji liczymy, jaki znak pojawił się tam najczęściej (z wagą pewności). Przykład dla pozycji 4: `KR123AB`, `KR123AB`, `KR12ZAB` daje `{"3": 1.8, "Z": 0.9}`, wygrywa `"3"`.

Dzięki temu nawet jeśli żaden pojedynczy odczyt nie jest idealny, wynik może być. `"".join(lista)` skleja listę znaków w jeden tekst.

```python
53     total = sum(reading.confidence for reading in readings)
54     agreeing = sum(reading.confidence for reading in readings if reading.text == plate)
55     return plate, agreeing / total
```

Pewność przejazdu to jaka część „głosów” była w całości zgodna z wynikiem. Dla przykładu wyżej: 1.8 / 2.7 = 0.67, czyli 67%. `sum(... for ...)` to wyrażenie generatorowe, jak listowe, tylko bez tworzenia listy w pamięci.

```python
58 def is_same_plate_position(previous: Box, current: Box) -> bool:
59     (px, py), (cx, cy) = previous.center, current.center
60     distance = math.hypot(cx - px, cy - py)
61     return distance < 1.5 * max(previous.width, current.width)
```

Czy tablica w nowej klatce to ta sama, co wcześniej?

- Linia 59 to **rozpakowanie krotek**: z dwóch środków wyciągamy cztery liczby naraz.
- `math.hypot(dx, dy)` to odległość z twierdzenia Pitagorasa: √(dx² + dy²).
- Próg to 1,5 szerokości tablicy. Taki próg sam dopasowuje się do skali: tablica blisko kamery jest duża i może przesunąć się o więcej pikseli, a tablica daleko jest mała.

```python
64 class PlateTracker:
65     def __init__(self, max_gap_seconds: float = 2.0, min_readings: int = 3, repeat_cooldown_seconds: float = 60.0) -> None:
66         self.max_gap_seconds = max_gap_seconds
67         self.min_readings = min_readings
68         self.repeat_cooldown_seconds = repeat_cooldown_seconds
69         self.tracks: list[Track] = []
70         self.last_emitted: dict[str, float] = {}
```

Trzy ustawienia:

| Ustawienie | Domyślnie | Znaczenie |
|---|---|---|
| `max_gap_seconds` | 2 s | jak długo auto może być niewidoczne (np. zasłonięte przez pieszego), zanim uznamy, że odjechało |
| `min_readings` | 3 | ślad z 1-2 odczytami to prawdopodobnie przypadkowy błysk detektora, więc go pomijamy |
| `repeat_cooldown_seconds` | 60 s | ten sam numer w ciągu minuty to to samo auto (np. stoi pod szlabanem), nie nowy wjazd |

- `self.tracks` to lista aktywnych śladów (aut widocznych teraz).
- `self.last_emitted` to **słownik** (`dict`): numer → kiedy ostatnio go zapisaliśmy. Np. `{"KR123AB": 20.4}`.

```python
72     def update(self, time: float, readings: list[PlateReading], frame: np.ndarray | None = None) -> list[Vehicle]:
73         for reading in readings:
74             track = self._find_track(reading.box)
75             if track is None:
76                 track = Track(first_seen=time, last_seen=time, last_box=reading.box)
77                 self.tracks.append(track)
78             track.add(time, reading, frame)
```

Wywoływane dla każdej analizowanej klatki. Dla każdego odczytu: znajdź pasujący ślad, a jak go nie ma, załóż nowy (nowe auto w kadrze). Potem dołóż odczyt.

Linia 76 to **argumenty nazwane** (`first_seen=time`). Kolejność nie ma znaczenia i od razu widać, co jest czym.

```python
80         finished = [track for track in self.tracks if time - track.last_seen > self.max_gap_seconds]
81         self.tracks = [track for track in self.tracks if track not in finished]
82         return self._to_vehicles(finished)
```

Ślady, które nie dostały odczytu od ponad 2 sekund, są zakończone: auto odjechało. Wyjmujemy je z aktywnych i zamieniamy na pojazdy. Metoda zwraca listę gotowych pojazdów, zwykle pustą.

```python
84     def flush(self) -> list[Vehicle]:
85         finished, self.tracks = self.tracks, []
86         return self._to_vehicles(finished)
```

Wywoływane na końcu filmu: kończy wszystkie ślady naraz, bo auto mogło jeszcze stać w kadrze, kiedy film się skończył. Linia 85 to przypisanie dwóch zmiennych naraz: `finished` dostaje obecną listę, a `self.tracks` nową pustą.

```python
88     def _find_track(self, box: Box) -> Track | None:
89         candidates = [track for track in self.tracks if is_same_plate_position(track.last_box, box)]
90         if not candidates:
91             return None
92         return min(candidates, key=lambda track: math.dist(track.last_box.center, box.center))
```

- Podkreślnik na początku nazwy (`_find_track`) to umowa: metoda wewnętrzna, nie do używania spoza klasy.
- `if not candidates:` sprawdza, czy lista jest pusta, bo pusta lista jest „fałszywa”.
- `min(lista, key=...)` wybiera element z najmniejszą wartością wyliczoną przez funkcję `key`. Tu: najbliższy ślad.
- `lambda track: ...` to **funkcja anonimowa** w jednej linii, odpowiednik `def odleglosc(track): return math.dist(...)`.

```python
94     def _to_vehicles(self, tracks: list[Track]) -> list[Vehicle]:
95         vehicles = []
96         for track in sorted(tracks, key=lambda track: track.first_seen):
97             if len(track.readings) < self.min_readings:
98                 continue
99             plate, confidence = vote(track.readings)
```

Dla każdego zakończonego śladu, w kolejności pojawienia się: odrzuć za krótkie i zagłosuj nad numerem. `sorted` zwraca nową, posortowaną listę i nie zmienia oryginału.

```python
100            previous_last_seen = self.last_emitted.get(plate)
101            self.last_emitted[plate] = track.last_seen
102            if previous_last_seen is not None and track.first_seen - previous_last_seen < self.repeat_cooldown_seconds:
103                continue
```

Ochrona przed podwójnym liczeniem:

- `dict.get(klucz)` zwraca wartość albo `None`, jeśli klucza nie ma. Zwykłe `dict[klucz]` rzuciłoby wtedy błąd.
- Linia 101 aktualizuje czas zawsze, także gdy za chwilę pominiemy ślad. Dzięki temu auto, które stoi długo i co chwilę znika za pieszym, będzie pomijane, dopóki nie zniknie na całą minutę.

```python
104            vehicles.append(
105                Vehicle(
106                    plate=plate,
...
111                    snapshot=track.snapshot,
112                )
113            )
114        return vehicles
```

Tworzymy niezmienny obiekt `Vehicle` z wynikiem głosowania i danymi śladu.

---

## anpr/schema.sql

Opis tabel w języku SQL. Plik wykonuje się przy każdym połączeniu z bazą, dlatego wszędzie jest `IF NOT EXISTS`: przy drugim uruchomieniu nic się nie dzieje i dane nie znikają.

```sql
CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    direction TEXT NOT NULL CHECK (direction IN ('wjazd', 'wyjazd')),
    source TEXT NOT NULL
);
```

Tabela kamer.

| Kolumna | Opis |
|---|---|
| `id INTEGER PRIMARY KEY` | **klucz główny**, unikalny numer wiersza. W SQLite taka kolumna sama nadaje kolejne numery 1, 2, 3... |
| `name TEXT NOT NULL UNIQUE` | nazwa kamery. `NOT NULL` oznacza, że jest obowiązkowa, a `UNIQUE`, że dwie kamery nie mogą mieć tej samej nazwy |
| `direction ... CHECK (...)` | **ograniczenie**: baza sama odrzuci wartość inną niż `wjazd` lub `wyjazd`, nawet gdyby kod miał błąd |
| `source` | skąd był obraz (plik lub adres strumienia) |

```sql
CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id),
    plate TEXT NOT NULL,
    confidence REAL NOT NULL,
    readings INTEGER NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    snapshot_path TEXT
);
```

Tabela przejazdów, czyli serce bazy.

- `camera_id ... REFERENCES cameras(id)` to **klucz obcy**: przejazd wskazuje na kamerę. Baza nie pozwoli zapisać przejazdu z nieistniejącą kamerą. Tak tabele łączą się w relacje, stąd nazwa „relacyjna baza danych”.
- `REAL` to liczba zmiennoprzecinkowa.
- Daty jako `TEXT`, bo SQLite nie ma osobnego typu daty. Zapisujemy je w formacie ISO (`2026-10-08T20:01:50`), który sortuje się alfabetycznie tak samo jak chronologicznie.
- `snapshot_path` nie ma `NOT NULL`, więc może być puste.

Dlaczego nazwa kamery nie jest zapisana w każdym przejeździe? To **normalizacja**: nazwa jest w jednym miejscu. Zmiana nazwy kamery to zmiana jednego wiersza, a nie tysięcy.

```sql
CREATE INDEX IF NOT EXISTS idx_passages_plate ON passages(plate);
CREATE INDEX IF NOT EXISTS idx_passages_first_seen ON passages(first_seen);
```

**Indeksy** działają jak skorowidz w książce. Wyszukiwanie po numerze tablicy albo sortowanie po dacie nie musi przeglądać całej tabeli. Przy 4 wierszach nie ma różnicy, przy milionie jest ogromna.

```sql
CREATE TABLE IF NOT EXISTS voivodeships (
    letter TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

INSERT OR IGNORE INTO voivodeships (letter, name) VALUES
    ('B', 'podlaskie'),
    ...
```

Słownik: pierwsza litera tablicy → województwo. `INSERT OR IGNORE` pomija wiersze, które już są (ten sam klucz główny), więc ponowne uruchomienie nie zdubluje danych.

---

## anpr/database.py

Cała komunikacja z SQLite. Reszta programu nie pisze SQL-a, tylko woła te funkcje.

```python
1  import sqlite3
2  from datetime import datetime
3  from pathlib import Path
4
5  SCHEMA_PATH = Path(__file__).with_name("schema.sql")
```

- `sqlite3` jest wbudowany w Pythona. SQLite to cała baza danych w jednym pliku, bez serwera.
- `__file__` to ścieżka bieżącego pliku (`.../anpr/database.py`). `.with_name("schema.sql")` podmienia nazwę pliku i daje `.../anpr/schema.sql`. Dzięki temu schemat znajdzie się bez względu na to, z jakiego katalogu uruchomisz program.

```python
8  def connect(path: str | Path) -> sqlite3.Connection:
9      connection = sqlite3.connect(path)
10     connection.row_factory = sqlite3.Row
11     connection.execute("PRAGMA foreign_keys = ON")
12     connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
13     return connection
```

- Linia 9 otwiera plik bazy, a jeśli go nie ma, tworzy nowy.
- Linia 10: wiersze wyników da się czytać po nazwie kolumny (`row["plate"]`), a nie tylko po numerze (`row[1]`).
- Linia 11: SQLite **domyślnie nie sprawdza** kluczy obcych. Trzeba to włączyć przy każdym połączeniu.
- Linia 12 czyta `schema.sql` i wykonuje wszystkie polecenia naraz. `encoding="utf-8"` jest potrzebne dla polskich znaków (`dolnośląskie`).

```python
16 def get_or_create_camera(connection: sqlite3.Connection, name: str, direction: str, source: str) -> int:
17     row = connection.execute("SELECT id FROM cameras WHERE name = ?", (name,)).fetchone()
18     if row is not None:
19         return row["id"]
```

Popularny wzorzec „pobierz albo utwórz”. Najpierw szukamy kamery po nazwie.

- **`?` to parametr.** Wartość podajemy osobno, w krotce. Nigdy nie sklejamy SQL-a z tekstem użytkownika (`f"... WHERE name = '{name}'"`), bo to otwiera drogę do **SQL injection**: ktoś mógłby podać nazwę `x'; DROP TABLE passages; --` i skasować dane. Z `?` baza traktuje wartość wyłącznie jako dane.
- `(name,)` to krotka jednoelementowa. **Przecinek jest konieczny**: `(name)` to tylko `name` w nawiasie.
- `.fetchone()` zwraca pierwszy wiersz albo `None`.

```python
20     cursor = connection.execute(
21         "INSERT INTO cameras (name, direction, source) VALUES (?, ?, ?)",
22         (name, direction, source),
23     )
24     connection.commit()
25     return cursor.lastrowid
```

Jeśli kamery nie ma, dodajemy ją.

- `commit()` **zatwierdza zmiany** na dysku. Bez tego po zamknięciu programu wiersz by zniknął. To część mechanizmu transakcji: zmiany są „na brudno”, dopóki ich nie zatwierdzisz.
- `cursor.lastrowid` to `id`, które baza nadała nowemu wierszowi.

```python
28 def save_passage(
29     connection: sqlite3.Connection,
...
37 ) -> int:
```

Przy wielu argumentach każdy idzie w osobnej linii, tak łatwiej czytać.

```python
38     cursor = connection.execute(
39         """
40         INSERT INTO passages (camera_id, plate, confidence, readings, first_seen, last_seen, snapshot_path)
41         VALUES (?, ?, ?, ?, ?, ?, ?)
42         """,
```

Potrójny cudzysłów `"""` to napis wielolinijkowy. Tu zawiera SQL. Siedem kolumn i siedem znaków `?`.

```python
43         (
44             camera_id,
...
48             first_seen.isoformat(timespec="seconds"),
49             last_seen.isoformat(timespec="seconds"),
50             snapshot_path,
51         ),
52     )
53     connection.commit()
54     return cursor.lastrowid
```

`isoformat(timespec="seconds")` zamienia datę na tekst `2026-10-08T20:01:50`, bez ułamków sekund. Przecinek po ostatnim elemencie jest dozwolony. Ułatwia dopisywanie i czytanie zmian w gicie.

```python
57 def latest_passages(connection: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
58     return connection.execute(
59         """
60         SELECT passages.first_seen, passages.plate, passages.confidence, passages.readings,
61                cameras.name AS camera, cameras.direction
62         FROM passages
63         JOIN cameras ON cameras.id = passages.camera_id
64         ORDER BY passages.first_seen DESC, passages.id DESC
65         LIMIT ?
66         """,
67         (limit,),
68     ).fetchall()
```

Ostatnie przejazdy, razem z nazwą kamery.

| Fragment | Znaczenie |
|---|---|
| `SELECT ...` | które kolumny chcemy |
| `passages.plate` | kolumna `plate` z tabeli `passages`. Przedrostek jest potrzebny, gdy dwie tabele mają kolumnę o tej samej nazwie (obie mają `id`) |
| `cameras.name AS camera` | **alias**: w wyniku kolumna nazywa się `camera` |
| `JOIN cameras ON cameras.id = passages.camera_id` | **złączenie**: do każdego przejazdu dołącz wiersz kamery, której `id` zgadza się z `camera_id` |
| `ORDER BY ... DESC` | sortuj malejąco, czyli najnowsze pierwsze. Drugi klucz (`id`) rozstrzyga remisy w tej samej sekundzie |
| `LIMIT ?` | najwyżej N wierszy |

`.fetchall()` pobiera wszystkie wiersze wyniku jako listę.

```python
71 def entries_per_hour(connection: sqlite3.Connection) -> list[sqlite3.Row]:
...
74         SELECT strftime('%Y-%m-%d %H:00', passages.first_seen) AS hour, COUNT(*) AS entries
75         FROM passages
76         JOIN cameras ON cameras.id = passages.camera_id
77         WHERE cameras.direction = 'wjazd'
78         GROUP BY hour
79         ORDER BY hour
```

Liczba wjazdów w każdej godzinie.

- `strftime('%Y-%m-%d %H:00', data)` przycina datę do pełnej godziny: `2026-10-08T20:01:50` → `2026-10-08 20:00`.
- `WHERE` filtruje wiersze. Liczymy tylko kamery wjazdowe.
- `GROUP BY hour` grupuje wiersze o tej samej godzinie, a `COUNT(*)` liczy wiersze w każdej grupie. To podstawa każdego raportu i statystyki w SQL.

```python
84 def entries_per_voivodeship(connection: sqlite3.Connection) -> list[sqlite3.Row]:
...
87         SELECT COALESCE(voivodeships.name, 'inne') AS voivodeship, COUNT(*) AS entries
88         FROM passages
89         JOIN cameras ON cameras.id = passages.camera_id
90         LEFT JOIN voivodeships ON voivodeships.letter = substr(passages.plate, 1, 1)
91         WHERE cameras.direction = 'wjazd'
92         GROUP BY voivodeship
93         ORDER BY entries DESC
```

Wjazdy według województwa. Tu są dwa rodzaje złączeń:

- `JOIN` (wewnętrzne) zostawia tylko wiersze, które mają parę w drugiej tabeli.
- `LEFT JOIN` zostawia **wszystkie** przejazdy, także te bez pary. Tablica dyplomatyczna czy wojskowa (`U...`) nie ma województwa, ale nie powinna zniknąć ze statystyki. Wtedy `voivodeships.name` jest `NULL`.
- `substr(plate, 1, 1)` to pierwsza litera numeru. W SQL liczymy od 1, nie od 0 jak w Pythonie.
- `COALESCE(a, 'inne')` daje `a`, a jeśli to `NULL`, to `'inne'`.

---

## anpr/pipeline.py

Pętla, która łączy wszystko: czyta wideo, oddaje klatki do rozpoznawania, przekazuje odczyty do trackera i zapisuje pojazdy.

```python
1  import sqlite3
2  import time as clock
3  from datetime import datetime, timedelta
4  from pathlib import Path
5
6  import cv2
7  import numpy as np
```

- `import time as clock`: moduł `time` importujemy pod inną nazwą, bo w pętli mamy zmienną `time` (sekunda nagrania). Bez aliasu zmienna przykryłaby moduł.
- `cv2` to OpenCV, biblioteka do obrazu i wideo.

```python
13 WINDOW_NAME = "ANPR - q zamyka"
14 BOX_COLOR = (0, 200, 0)
```

Tytuł okna podglądu i kolor ramki. `(0, 200, 0)` w BGR to zielony.

```python
17 def draw_readings(frame: np.ndarray, readings: list[PlateReading]) -> np.ndarray:
18     annotated = frame.copy()
19     for reading in readings:
20         box = reading.box
21         cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), BOX_COLOR, 2)
22         label = f"{reading.text} {reading.confidence:.0%}"
23         cv2.putText(annotated, label, (box.x1, box.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2)
24     return annotated
```

Rysuje ramki na kopii klatki. Oryginał zostaje czysty, bo to on może trafić do bazy jako zdjęcie.

- `cv2.rectangle(obraz, róg1, róg2, kolor, grubość)`.
- `f"..."` to **f-string**: w klamrach wstawiamy wartości. `:.0%` formatuje liczbę jako procent bez miejsc po przecinku: `0.94` → `94%`.
- `cv2.putText(obraz, tekst, pozycja, czcionka, skala, kolor, grubość)`. Pozycja `box.y1 - 10` to 10 px nad ramką. Wbudowane czcionki OpenCV nie mają polskich znaków.

```python
27 def save_snapshot(vehicle: Vehicle, snapshots_dir: Path, camera_id: int, seen_at: datetime) -> str | None:
28     if vehicle.snapshot is None:
29         return None
30     snapshots_dir.mkdir(parents=True, exist_ok=True)
31     path = snapshots_dir / f"{camera_id}_{seen_at:%Y%m%d_%H%M%S}_{vehicle.plate}.jpg"
32     cv2.imwrite(str(path), vehicle.snapshot)
33     return str(path)
```

Zapisuje zdjęcie auta na dysk.

- `mkdir(parents=True, exist_ok=True)` tworzy folder razem z brakującymi folderami nadrzędnymi i nie zgłasza błędu, jeśli już istnieje.
- `snapshots_dir / "plik.jpg"`: obiekty `Path` łączy się operatorem `/`. Działa poprawnie na każdym systemie.
- `{seen_at:%Y%m%d_%H%M%S}` formatuje datę w f-stringu: `20261008_200150`. Nazwa pliku to np. `1_20261008_200150_KR123AB.jpg`.
- `cv2.imwrite` przyjmuje ścieżkę jako zwykły tekst, stąd `str(path)`.

```python
36 def store_vehicle(
...
42 ) -> None:
43     first_seen = started_at + timedelta(seconds=vehicle.first_seen)
44     last_seen = started_at + timedelta(seconds=vehicle.last_seen)
```

Tracker liczy czas w sekundach nagrania. Tu zamieniamy go na prawdziwą datę: początek analizy + przesunięcie. `timedelta` to odstęp czasu, który da się dodać do daty.

```python
45     snapshot_path = save_snapshot(vehicle, snapshots_dir, camera_id, first_seen)
46     database.save_passage(
...
55     )
56     print(f"{first_seen:%H:%M:%S}  {vehicle.plate:<9} pewność {vehicle.confidence:.0%}  odczytów {vehicle.readings}")
```

Zapis zdjęcia, zapis do bazy i komunikat w terminalu. `:<9` wyrównuje tekst do lewej w polu szerokości 9 znaków, żeby kolumny w terminalu były równe.

```python
59 def process_source(
60     source: str,
...
64     every: int = 3,
65     show: bool = False,
66     snapshots_dir: Path = Path("data/snapshots"),
67 ) -> list[Vehicle]:
```

Główna funkcja. `every=3` to analiza co trzeciej klatki: przy 25 klatkach na sekundę daje ok. 8 analiz na sekundę, wystarczająco, a 3 razy taniej.

```python
68     capture = cv2.VideoCapture(source)
69     if not capture.isOpened():
70         raise RuntimeError(f"Nie udało się otworzyć źródła wideo: {source}")
```

`VideoCapture` otwiera plik albo strumień sieciowy (`rtsp://...`). Jeśli się nie da, **zgłaszamy wyjątek** (`raise`): program zatrzyma się z czytelnym komunikatem, zamiast działać dalej na pustym źródle.

```python
72     fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
```

Liczba klatek na sekundę z nagrania. Niektóre strumienie jej nie podają i wtedy `get` zwraca `0`. `0 or 25.0` daje `25.0`, bo `or` zwraca pierwszą „prawdziwą” wartość, a `0` jest „fałszywe”.

```python
73     started_at = datetime.now()
74     started_clock = clock.monotonic()
```

Dwa zegary:

- `datetime.now()` to data i godzina do zapisu w bazie.
- `clock.monotonic()` to licznik sekund, który zawsze rośnie, nawet gdy system zmieni godzinę (np. zmiana czasu na zimowy). Służy do mierzenia upływu czasu w podglądzie.

```python
75     tracker = PlateTracker()
76     vehicles: list[Vehicle] = []
77     readings: list[PlateReading] = []
78     frame_index = 0
```

Stan pętli: tracker, lista zapisanych pojazdów, ostatnie odczyty (do rysowania) i numer klatki.

```python
80     try:
81         while True:
82             ok, frame = capture.read()
83             if not ok:
84                 break
```

- `try: ... finally: ...`: kod w `finally` wykona się **zawsze**, także po błędzie albo Ctrl+C. Dzięki temu kamera zostanie zwolniona.
- `while True:` to pętla „bez końca”, przerywana przez `break`.
- `capture.read()` zwraca dwie wartości: czy się udało i klatkę. `ok` jest `False` na końcu pliku albo po zerwaniu strumienia.

```python
85             time = frame_index / fps
```

Sekunda nagrania, np. klatka 50 przy 25 FPS to 2.0 s. Liczymy ją z numeru klatki, a nie z zegara, więc wynik jest **powtarzalny**: ten sam film zawsze daje ten sam wynik, niezależnie od szybkości komputera.

```python
87             if frame_index % every == 0:
88                 readings = recognizer.read(frame)
89                 for vehicle in tracker.update(time, readings, frame):
90                     store_vehicle(connection, camera_id, vehicle, started_at, snapshots_dir)
91                     vehicles.append(vehicle)
```

- `%` to **reszta z dzielenia**. `frame_index % 3 == 0` jest prawdą dla klatek 0, 3, 6, 9...
- Odczyty idą do trackera. Jeśli jakieś auto właśnie „odjechało”, dostajemy je z powrotem i zapisujemy.

```python
93             if show:
94                 cv2.imshow(WINDOW_NAME, draw_readings(frame, readings))
95                 delay_ms = int((time - (clock.monotonic() - started_clock)) * 1000)
96                 if cv2.waitKey(max(1, delay_ms)) & 0xFF == ord("q"):
97                     break
```

Podgląd w oknie (tylko z `--show`).

- Linia 94 rysuje ramki i pokazuje klatkę. Między analizami rysujemy ostatnie znane odczyty, więc ramka nie miga.
- Linia 95: ile milisekund nagranie „wyprzedza” prawdziwy czas. Program liczy szybciej, niż film trwa, więc czekamy, żeby film leciał w normalnym tempie.
- `cv2.waitKey(ms)` czeka podaną liczbę milisekund, obsługuje okno (bez tego okno się nie odświeży) i zwraca kod wciśniętego klawisza. `max(1, ...)` pilnuje, żeby czekać co najmniej 1 ms.
- `& 0xFF` zostawia tylko ostatni bajt kodu klawisza (na niektórych systemach są tam dodatkowe bity). `ord("q")` to kod litery q (113).

```python
98             frame_index += 1
99     finally:
100        capture.release()
101        if show:
102            cv2.destroyAllWindows()
```

`+= 1` to skrót od `frame_index = frame_index + 1`. W `finally` zwalniamy plik albo kamerę i zamykamy okno.

```python
104    for vehicle in tracker.flush():
105        store_vehicle(connection, camera_id, vehicle, started_at, snapshots_dir)
106        vehicles.append(vehicle)
107    return vehicles
```

Koniec filmu: auta, które jeszcze były w kadrze, też trzeba zapisać.

---

## anpr/\_\_main\_\_.py

Interfejs wiersza poleceń. Plik o nazwie `__main__.py` w pakiecie uruchamia się poleceniem `python -m anpr`.

```python
1  import argparse
2  import logging
3
4  import onnxruntime
```

`argparse` to wbudowany moduł do obsługi argumentów (`--show`, `--camera`). Sam generuje pomoc (`python -m anpr run --help`).

```python
11 DEFAULT_DB = "data/anpr.db"
```

Domyślna ścieżka bazy.

```python
14 def run(args: argparse.Namespace) -> None:
15     connection = database.connect(args.db)
16     camera_id = database.get_or_create_camera(connection, args.camera, args.direction, args.source)
17     recognizer = PlateRecognizer()
18     print(f"Kamera: {args.camera} ({args.direction}), źródło: {args.source}")
19     vehicles = process_source(args.source, connection, camera_id, recognizer, every=args.every, show=args.show)
20     print(f"Zapisano przejazdów: {len(vehicles)}")
```

Polecenie `run`: połącz z bazą, znajdź albo dodaj kamerę, załaduj modele, przetwórz źródło i wypisz podsumowanie. `args` to obiekt z argumentami z linii poleceń: `args.camera` to wartość `--camera`.

```python
23 def report(args: argparse.Namespace) -> None:
24     connection = database.connect(args.db)
25     show_plate = (lambda plate: plate) if args.full else mask
```

W Pythonie funkcję można zapisać w zmiennej jak każdą wartość. Z `--full` `show_plate` zwraca numer bez zmian, a bez tej opcji to funkcja `mask`. Potem po prostu wołamy `show_plate(numer)` i nie trzeba pisać `if` w każdej linii.

`a if warunek else b` to **wyrażenie warunkowe**, czyli `if` w jednej linii.

```python
27     print("Ostatnie przejazdy")
28     for row in database.latest_passages(connection, args.limit):
29         print(f"  {row['first_seen']}  {show_plate(row['plate']):<9} {row['confidence']:.0%}  {row['camera']} ({row['direction']})")
```

Dzięki `row_factory = sqlite3.Row` piszemy `row['plate']`. Wewnątrz f-stringu w podwójnym cudzysłowie używamy pojedynczych cudzysłowów.

```python
31     print("\nWjazdy na godzinę")
```

`\n` to znak nowej linii, czyli pusta linia przed nagłówkiem.

```python
40 def main() -> None:
41     logging.disable(logging.WARNING)
42     onnxruntime.set_default_logger_severity(3)
```

Wyciszamy komunikaty techniczne bibliotek: `logging.disable(WARNING)` wyłącza ostrzeżenia i informacje, a w ONNX Runtime poziom 3 oznacza, że pokazuje tylko błędy.

```python
43     parser = argparse.ArgumentParser(prog="anpr", description="...")
44     parser.add_argument("--db", default=DEFAULT_DB, help="ścieżka do pliku bazy SQLite")
45     commands = parser.add_subparsers(required=True)
```

- Linia 44: opcja wspólna dla wszystkich poleceń. Podaje się ją **przed** nazwą polecenia: `python -m anpr --db data/demo.db run ...`.
- Linia 45: **podpolecenia** (`run`, `report`), jak `git commit` i `git push`. `required=True` sprawia, że trzeba podać któreś.

```python
47     run_parser = commands.add_parser("run", help="analizuj film lub strumień z kamery")
48     run_parser.add_argument("source", help="...")
49     run_parser.add_argument("--camera", required=True, help="...")
50     run_parser.add_argument("--direction", choices=["wjazd", "wyjazd"], default="wjazd")
51     run_parser.add_argument("--every", type=int, default=3, help="...")
52     run_parser.add_argument("--show", action="store_true", help="...")
53     run_parser.set_defaults(handler=run)
```

| Linia | Co robi |
|---|---|
| 48 | `source` bez `--` to argument **pozycyjny**, czyli obowiązkowy, podawany bez nazwy |
| 49 | `required=True`: opcja z `--`, ale obowiązkowa |
| 50 | `choices`: dozwolone tylko te wartości. `argparse` sam zgłosi błąd przy innej |
| 51 | `type=int`: zamienia tekst z terminala na liczbę |
| 52 | `store_true`: przełącznik bez wartości. Jest `--show`, to `True`, a nie ma, to `False` |
| 53 | sztuczka: do argumentów dopisujemy, którą funkcję wywołać dla tego polecenia |

```python
60     args = parser.parse_args()
61     args.handler(args)
```

Odczytaj argumenty z terminala i wywołaj odpowiednią funkcję: `run(args)` albo `report(args)`. Dzięki linii 53 nie potrzeba `if args.command == "run": ...`.

```python
64 if __name__ == "__main__":
65     main()
```

Klasyczny idiom Pythona. `__name__` ma wartość `"__main__"` tylko wtedy, gdy plik jest uruchamiany bezpośrednio. Przy imporcie z innego pliku `main()` się nie wykona.

---

## anpr/\_\_init\_\_.py

Pusty plik. Oznacza, że folder `anpr` to **pakiet** Pythona, czyli zbiór modułów, który można importować: `from anpr import database`.

---

## scripts/evaluate.py

Mierzy skuteczność na 22 nagraniach AGH, porównując wyniki z plikiem poprawnych numerów. Uruchamiany jako `python -m scripts.evaluate`, żeby Python widział pakiet `anpr` z katalogu projektu.

```python
12 VIDEOS_DIR = Path("data/videos")
13 GROUND_TRUTH = Path("data/ground_truth.csv")
14 EVALUATION_DB = Path("data/evaluation.db")
```

**Ground truth** to „prawda wzorcowa”, czyli poprawne odpowiedzi przygotowane przez ludzi. Ocena ma osobną bazę, żeby nie mieszać jej z danymi z `run`.

```python
17 def load_ground_truth() -> dict[str, str]:
18     with GROUND_TRUTH.open(encoding="utf-8") as file:
19         return {row["SRC"]: row["Vehicle_registration_plate"] for row in csv.DictReader(file, delimiter=";")}
```

- `with ... as file:` to **menedżer kontekstu**: plik zostanie zamknięty automatycznie po wyjściu z bloku, nawet przy błędzie.
- `csv.DictReader` czyta CSV i każdy wiersz zwraca jako słownik z nagłówkami jako kluczami: `{"SRC": "src1", "Number_of_characters": "7", ...}`. `delimiter=";"`, bo AGH rozdziela kolumny średnikiem, a nie przecinkiem.
- `{klucz: wartość for ...}` to **wyrażenie słownikowe**. Wynik to np. `{"src1": "...", "src4": "..."}`.

```python
22 def main() -> None:
...
25     EVALUATION_DB.unlink(missing_ok=True)
26     connection = database.connect(EVALUATION_DB)
27     recognizer = PlateRecognizer()
28     expected_plates = load_ground_truth()
```

`unlink` usuwa plik starej bazy, więc każdy pomiar zaczyna od zera. `missing_ok=True` sprawia, że brak pliku nie jest błędem. Modele ładujemy raz, dla wszystkich 22 filmów.

```python
30     correct = 0
31     started = time.perf_counter()
32     for name, expected in expected_plates.items():
33         video = VIDEOS_DIR / f"agh_{name}.avi"
34         camera_id = database.get_or_create_camera(connection, f"AGH {name}", "wjazd", str(video))
35         vehicles = process_source(str(video), connection, camera_id, recognizer, snapshots_dir=Path("data/snapshots/evaluation"))
```

- `time.perf_counter()` to najdokładniejszy zegar do mierzenia czasu wykonania.
- `.items()` daje pary klucz-wartość ze słownika. Rozpakowujemy je od razu do `name` i `expected`.
- Każdy film to osobna kamera. Gdyby wszystkie były jedną kamerą, ochrona przed powtórzeniem w ciągu 60 s mogłaby pomieszać wyniki.

```python
36         plates = [vehicle.plate for vehicle in vehicles]
37         is_correct = plates == [expected]
38         correct += is_correct
```

- Wynik jest poprawny tylko wtedy, gdy lista zapisanych numerów to **dokładnie jeden, właściwy numer**. To łapie zarówno błędny odczyt, jak i podwójny zapis (jak pierwotny błąd w src26).
- `correct += is_correct` działa, bo w Pythonie `True` to 1, a `False` to 0.

```python
39         print(f"{'OK ' if is_correct else 'BŁĄD'} {name:<6} oczekiwano {expected:<9} odczytano {', '.join(plates) or '-'}")
```

`', '.join(plates) or '-'`: jeśli lista jest pusta, `join` da pusty tekst, który jest „fałszywy”, więc `or` podstawi `-`.

```python
41     elapsed = time.perf_counter() - started
42     print(f"\nSkuteczność: {correct}/{len(expected_plates)} ({correct / len(expected_plates):.0%}), czas {elapsed:.0f} s")
```

Podsumowanie: liczba trafień, procent i czas. `:.0f` to liczba bez miejsc po przecinku.

---

## tests/

Testy automatyczne uruchamiane przez **pytest** (`.venv/bin/python -m pytest -q`). Pytest sam znajduje pliki `test_*.py` i funkcje `test_*`, uruchamia je i raportuje, które przeszły. `assert warunek` mówi: „to musi być prawdą, inaczej test nie przechodzi”.

Testy używają zmyślonych numerów, żeby prawdziwe tablice z nagrań nie trafiły do repozytorium.

Po co testy? Po każdej zmianie w kodzie jednym poleceniem sprawdzasz w 0,3 s, czy nic się nie zepsuło. Bez nich trzeba by za każdym razem oglądać filmy.

### tests/test_plates.py

| Test | Co sprawdza |
|---|---|
| `test_normalize_removes_spaces_and_symbols` | `"kr 123-ab"` → `"KR123AB"` |
| `test_accepts_polish_plates` | 4 różne poprawne formaty (7 i 8 znaków, wyróżnik jedno-, dwu- i trzyliterowy) są akceptowane |
| `test_rejects_garbage` | pusty tekst, za krótki, same cyfry i za długi są odrzucane |
| `test_mask_keeps_district_prefix` | maskowanie zostawia 3 pierwsze znaki |

### tests/test_tracker.py

```python
4  BOX = Box(100, 100, 200, 130)
7  def reading(text: str, confidence: float = 0.9, box: Box = BOX) -> PlateReading:
8      return PlateReading(text, confidence, box)
```

Funkcja pomocnicza: szybkie tworzenie odczytu w testach, z sensownymi wartościami domyślnymi. Test podaje tylko to, co dla niego ważne.

| Test | Co sprawdza |
|---|---|
| `test_vote_fixes_single_wrong_character` | 2 dobre odczyty i 1 z błędnym znakiem dają dobry numer z pewnością 67% |
| `test_vote_ignores_truncated_readings` | obcięty odczyt (6 znaków zamiast 7) nie psuje wyniku |
| `test_one_vehicle_is_emitted_after_it_disappears` | 5 odczytów w sekundach 0-4: do tego czasu nic nie wychodzi. W sekundzie 10 (6 s przerwy) dostajemy dokładnie jeden pojazd z 5 odczytami |
| `test_short_flicker_is_ignored` | pojedynczy odczyt to nie samochód |
| `test_two_plates_far_apart_are_two_vehicles` | dwie tablice daleko od siebie w tej samej klatce to dwa auta |
| `test_same_plate_soon_after_is_not_counted_twice` | ten sam numer 5 s po zniknięciu nie daje drugiego wpisu |

`sorted(...)` w teście z dwoma autami sprawia, że kolejność nie ma znaczenia dla wyniku.

### tests/test_database.py

```python
7      connection = database.connect(":memory:")
```

`":memory:"` to specjalna nazwa SQLite: baza tylko w pamięci RAM, znika po teście. Szybka i nie zostawia plików.

```python
8      entry = database.get_or_create_camera(connection, "Brama", "wjazd", "film.avi")
9      exit_ = database.get_or_create_camera(connection, "Wyjazd", "wyjazd", "film2.avi")
10     assert database.get_or_create_camera(connection, "Brama", "wjazd", "film.avi") == entry
```

Dwie kamery. `exit_` ma podkreślnik, bo `exit` to wbudowana funkcja Pythona i lepiej jej nie przykrywać. Linia 10 sprawdza, że drugie wywołanie z tą samą nazwą nie tworzy duplikatu, tylko zwraca to samo `id`.

```python
12     seen = datetime(2026, 10, 8, 8, 15)
13     for plate in ["KR123AB", "KR456CD", "WA12345"]:
14         database.save_passage(connection, entry, plate, 0.9, 10, seen, seen, None)
15     database.save_passage(connection, exit_, "XX12345", 0.9, 10, seen, seen, None)
```

3 wjazdy (2 z K, 1 z W) i 1 wyjazd.

```python
17     assert len(database.latest_passages(connection)) == 4
18     assert [tuple(row) for row in database.entries_per_hour(connection)] == [("2026-10-08 08:00", 3)]
19     assert [tuple(row) for row in database.entries_per_voivodeship(connection)] == [
20         ("małopolskie", 2),
21         ("mazowieckie", 1),
22     ]
```

- Lista przejazdów zawiera wszystkie 4.
- Statystyki wjazdów liczą tylko 3, bo wyjazd jest odfiltrowany przez `WHERE direction = 'wjazd'`.
- `tuple(row)` zamienia wiersz bazy na zwykłą krotkę, żeby dało się ją porównać z oczekiwaną wartością.

---

## requirements.txt

```
fast-alpr[onnx]==0.4.0
opencv-python==5.0.0.93
pytest
```

Lista bibliotek do zainstalowania (`uv pip install -r requirements.txt`).

- `==0.4.0` **przypina wersję**. Za rok instalacja da dokładnie to samo, a nowa wersja biblioteki nie zepsuje projektu po cichu.
- `[onnx]` to dodatek (extra): razem z `fast-alpr` instaluje się ONNX Runtime.
- `opencv-python` to OpenCV z obsługą okien (potrzebne do `--show`).
- `pytest` bez wersji, bo to narzędzie tylko do testów.

---

## .gitignore

Lista plików, których git ma nie dodawać do repozytorium.

| Wpis | Dlaczego |
|---|---|
| `.venv/` | środowisko z bibliotekami, ok. 230 MB. Każdy odtwarza je z `requirements.txt` |
| `__pycache__/`, `.pytest_cache/` | pliki tymczasowe tworzone automatycznie przez Pythona i pytest |
| `*.md`, `.claude/` | wszystkie pliki Markdown (notatki, objaśnienia) i ustawienia asystenta, prywatne |
| `data/` | nagrania, bazy, zdjęcia aut i CSV z numerami tablic. **Dane osobowe (RODO)** i duże pliki |
| `snapshots/` | zdjęcia aut, gdyby kiedyś trafiły poza `data/` |
| `*.db`, `*.db-journal`, `*.db-wal`, `*.db-shm`, `*.sqlite`, `*.sqlite3` | bazy SQLite i ich pliki pomocnicze w dowolnym miejscu |
| `*.avi`, `*.mp4`, `*.mkv`, `*.mov` | nagrania wideo w dowolnym miejscu |
| `*.log` | logi mogą zawierać odczytane numery |
| `.env`, `.env.*` | przyszłe hasła, np. adres kamery IP z loginem `rtsp://user:haslo@...` |
| `!.env.example` | wyjątek (`!`): wzór pliku `.env` bez wartości ma trafić do repo |
| `.DS_Store` | śmieci systemu macOS |

---

## Słowniczek pojęć Pythona

| Pojęcie | Przykład w projekcie | W jednym zdaniu |
|---|---|---|
| moduł | `plates.py` | plik `.py`, który można importować |
| pakiet | `anpr/` | folder modułów z `__init__.py` |
| funkcja | `normalize` | nazwany kawałek kodu z wejściem i wyjściem |
| klasa | `PlateTracker` | przepis na obiekty łączące dane i funkcje |
| obiekt | `tracker = PlateTracker()` | konkretny egzemplarz klasy |
| metoda | `tracker.update(...)` | funkcja należąca do klasy, dostaje `self` |
| dataclass | `Box`, `Vehicle` | klasa do przechowywania danych, bez pisania konstruktora |
| adnotacja typu | `text: str -> str` | podpowiedź, jakiego typu są dane. Python jej nie wymusza |
| lista | `[1, 2, 3]` | uporządkowany, zmienny zbiór elementów |
| krotka | `(x, y)` | jak lista, ale niezmienna |
| słownik | `{"KR123AB": 20.4}` | pary klucz → wartość |
| wyrażenie listowe | `[r for r in readings if ...]` | krótki zapis pętli budującej listę |
| lambda | `lambda t: t.first_seen` | jednolinijkowa funkcja bez nazwy |
| f-string | `f"{x:.0%}"` | tekst z wstawionymi wartościami |
| wyjątek | `raise RuntimeError(...)` | zgłoszenie błędu, które przerywa program, jeśli nikt go nie obsłuży |
| `try` / `finally` | `pipeline.py` | `finally` wykona się zawsze, nawet po błędzie |
| `with` | `evaluate.py` | automatyczne sprzątanie (np. zamknięcie pliku) |
| `None` | `return None` | „brak wartości” |

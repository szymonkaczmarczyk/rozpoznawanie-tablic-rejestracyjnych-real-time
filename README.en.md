# License plate recognition

[Polski](README.md) · **English**

> Video goes in, vehicle passages come out.

An ANPR (automatic number plate recognition) system in Python. It reads video from a file or a network camera,
finds Polish license plates in the frames, reads the number, merges all readings of one car into a single
passage and stores it in SQLite together with a snapshot. The database gives simple stats right away: latest
passages, entries per hour and entries per voivodeship (Polish region).

The project is also a way to learn Python and SQL, so the database layer is plain SQL through `sqlite3` with
no ORM, and the whole codebase is explained line by line in [`docs/jak-dziala-kod.md`](docs/jak-dziala-kod.md)
(Polish).

**Result:** 22 out of 22 plates read correctly on the test recordings from the AGH parking lot, processed about
6 times faster than real time on an Apple M4 Mac (CPU only).

## Who it is for

- **People learning computer vision** who want to see a complete system, from a video frame to a database
  row, not just a single model.
- **Owners of a parking lot, gate or housing estate** who want to check whether their camera is good enough
  for plate reading before buying a commercial system.
- **Recruiters and developers** - small, readable code (about 450 lines) with tests and documentation.

What it is **not**: a ready-made access control system. It does not open barriers, has no web panel and no
allow list.

## How it works

```
video file or stream (rtsp://, http://)
   │
   ▼
pipeline.py      reads frames, sends every 3rd frame for analysis
   │
   ▼
recognizer.py    YOLO finds the plate, OCR reads the text, plates.py drops garbage
   │
   ▼
tracker.py       groups readings into one car's track and votes on the number
   │
   ▼
database.py      saves the passage to SQLite and the snapshot to data/snapshots
```

1. **Detection and OCR** - the [fast-alpr](https://github.com/ankandrew/fast-alpr) library with ONNX models:
   the `yolo-v9-t-384-license-plate-end2end` detector and the `cct-xs-v2-global-model` OCR. Runs on ONNX
   Runtime on the CPU, about 13 ms per frame.
2. **Filtering** - plates narrower than 80 px are skipped because OCR starts losing characters. The number
   has to match the Polish pattern: 4-8 characters starting with 1-3 letters.
3. **Tracking** - a reading joins an existing track when the plate center is closer than 1.5 plate widths.
   A track ends after 2 s without readings and needs at least 3 readings.
4. **Voting** - the most common number length wins first, then a confidence-weighted majority character by
   character. Passage confidence is the share of readings that agree with the result.
5. **Saving** - the same number on the same camera within 60 s is not counted twice.

## Features

- Video from a file or any stream OpenCV can open (`rtsp://`, `http://.../stream.m3u8`).
- Live preview with boxes and readings (`--show`), played at the recording's speed.
- SQLite database: cameras, passages, voivodeships by the plate's first letter.
- Terminal report with numbers masked by default (`KR1****`).
- Evaluation script for recordings with a ground truth file.
- 11 unit tests (pytest) using made-up plate numbers.

Sample report (made-up data):

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

The command line output is in Polish: latest passages, entries per hour, entries per voivodeship.

## Getting started

Requires Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Analyze a video and print the report:

```bash
.venv/bin/python -m anpr run data/videos/film.avi --camera "Gate" --show
.venv/bin/python -m anpr report
```

| Command | What it does |
|---|---|
| `anpr run SOURCE --camera NAME` | Analyzes a video or stream and saves passages |
| `--direction wjazd\|wyjazd` | Camera direction stored in the database, entry or exit (default `wjazd`) |
| `--every N` | Analyzes every Nth frame (default 3) |
| `--show` | Opens a preview window, `q` closes it |
| `anpr report` | Latest passages and stats |
| `anpr report --full` | Same, with full plate numbers |
| `--db PATH` | Another database file (default `data/anpr.db`) |
| `python -m scripts.evaluate` | Accuracy on recordings in `data/videos` and `data/ground_truth.csv` |

ONNX models are downloaded automatically on the first run.

## Test data

Accuracy was measured on recordings from the
[AGH Parking Database](https://qoe.agh.edu.pl/parking-database/): 22 videos, 1280x720, 25 FPS, one car each.
The recordings are not included in the repository. If you publish results, cite the database page and the paper
by Leszczuk et al. (2011).

For reliable reading the plate should be at least 80-100 px wide in the frame and the camera should look at
less than 30° to the driving direction. Most public panoramic webcams do not have enough resolution for that.

## Structure

```
anpr/
  plates.py       plate normalization, validation and masking
  recognizer.py   detection and OCR of a single frame
  tracker.py      plate tracking across frames and voting
  database.py     saving passages and SQL queries
  schema.sql      tables: cameras, passages, voivodeships
  pipeline.py     video loop, saving passages and snapshots
  __main__.py     command line (run, report)
scripts/
  evaluate.py     accuracy on test recordings
tests/            unit tests
docs/
  jak-dziala-kod.md   line-by-line code walkthrough (Polish)
data/             recordings, databases and snapshots (not in git)
```

## Privacy

A plate number combined with a place and time can be personal data under GDPR. That is why:

- databases, recordings and car snapshots are in `.gitignore` and never reach the repository,
- the report masks numbers by default,
- tests use made-up numbers only.

Use the program only on footage you have the right to process, and keep the data for a short time.

## Roadmap

- A virtual line in the frame instead of a direction set per camera.
- Longer tests on live streams (RTSP, HLS, YouTube Live).
- A Streamlit dashboard: passage list, entries chart, search.
- Fine-tuning the detector on Polish plates.

## Documentation

- [`docs/jak-dziala-kod.md`](docs/jak-dziala-kod.md) - every file explained line by line, the path of a frame
  through the program and a glossary of Python terms (Polish).

## License

[MIT](LICENSE) © 2026 Szymon Kaczmarczyk

The fast-alpr library and its models are MIT licensed. The AGH recordings have their own terms of use.

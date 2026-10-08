CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    direction TEXT NOT NULL CHECK (direction IN ('wjazd', 'wyjazd')),
    source TEXT NOT NULL
);

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

CREATE INDEX IF NOT EXISTS idx_passages_plate ON passages(plate);
CREATE INDEX IF NOT EXISTS idx_passages_first_seen ON passages(first_seen);

CREATE TABLE IF NOT EXISTS voivodeships (
    letter TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

INSERT OR IGNORE INTO voivodeships (letter, name) VALUES
    ('B', 'podlaskie'),
    ('C', 'kujawsko-pomorskie'),
    ('D', 'dolnośląskie'),
    ('E', 'łódzkie'),
    ('F', 'lubuskie'),
    ('G', 'pomorskie'),
    ('K', 'małopolskie'),
    ('L', 'lubelskie'),
    ('N', 'warmińsko-mazurskie'),
    ('O', 'opolskie'),
    ('P', 'wielkopolskie'),
    ('R', 'podkarpackie'),
    ('S', 'śląskie'),
    ('T', 'świętokrzyskie'),
    ('W', 'mazowieckie'),
    ('Z', 'zachodniopomorskie');

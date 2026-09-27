"""Steps 4 and 7 of the tile flow: raw tiles in a content-addressed folder, records in SQLite - plus the queries behind
the "querying the results" levels of DESIGN_NOTE.md §4.3 (lookup, filter, aggregate, review queue, CSV export).

* tiles        one row per unique image (id = SHA-256 of the uploaded bytes)
* models       one row per model artifact that has served predictions
* predictions  append-only: a new model version, or a retry after a failure, adds a row instead of overwriting
* reviews      append-only analyst labels (the ongoing ground truth); the latest one per tile counts
* current_predictions / current_reviews (views) the latest prediction per (tile, model version) / review per tile
"""
import json
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_version TEXT PRIMARY KEY,
    classes       TEXT NOT NULL,              -- JSON list, index order
    onnx_sha256   TEXT NOT NULL,
    manifest      TEXT NOT NULL,              -- full manifest.json, so any stored result can be explained later
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tiles (
    tile_id       TEXT PRIMARY KEY,           -- sha256 of the uploaded bytes
    source_name   TEXT,
    received_at   TEXT NOT NULL,
    captured_at   TEXT,
    lat           REAL,
    lon           REAL,
    sensor        TEXT,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    bands         INTEGER NOT NULL,
    byte_size     INTEGER NOT NULL,
    quality_flags TEXT NOT NULL,              -- JSON list
    blob_path     TEXT NOT NULL               -- relative to the blob directory
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id        TEXT NOT NULL REFERENCES tiles(tile_id),
    model_version  TEXT NOT NULL REFERENCES models(model_version),
    policy_version TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('confident', 'needs_review', 'unfamiliar', 'failed')),
    label          TEXT,
    confidence     REAL,
    margin         REAL,
    probabilities  TEXT,                      -- JSON {class: p}, all classes, calibrated
    ood_score      REAL,
    latency_ms     REAL,
    error          TEXT,
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id    TEXT NOT NULL REFERENCES tiles(tile_id),
    label      TEXT NOT NULL,                 -- one of the model's classes, or Other / Unusable
    note       TEXT,
    reviewer   TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_predictions_tile  ON predictions (tile_id, model_version, prediction_id);
CREATE INDEX IF NOT EXISTS ix_predictions_query ON predictions (model_version, status, label, confidence);
CREATE INDEX IF NOT EXISTS ix_reviews_tile      ON reviews (tile_id, review_id);
CREATE VIEW IF NOT EXISTS current_predictions AS
    SELECT p.* FROM predictions p
    WHERE p.prediction_id = (SELECT MAX(q.prediction_id) FROM predictions q
                             WHERE q.tile_id = p.tile_id AND q.model_version = p.model_version);
CREATE VIEW IF NOT EXISTS current_reviews AS
    SELECT r.* FROM reviews r
    WHERE r.review_id = (SELECT MAX(s.review_id) FROM reviews s WHERE s.tile_id = r.tile_id);
"""

FROM_SQL = """
FROM current_predictions p JOIN tiles t USING (tile_id) LEFT JOIN current_reviews r USING (tile_id)
"""
RESULT_SQL = """
SELECT p.prediction_id, p.tile_id, p.model_version, p.policy_version, p.status, p.label, p.confidence, p.margin,
       p.probabilities, p.ood_score, p.latency_ms, p.error, p.created_at,
       t.source_name, t.received_at, t.captured_at, t.lat, t.lon, t.sensor, t.quality_flags, t.byte_size,
       r.label AS review_label, r.reviewer AS reviewer, r.created_at AS reviewed_at
""" + FROM_SQL

SORTS = {
    'newest': 'p.prediction_id DESC', 'oldest': 'p.prediction_id ASC',
    'confidence_asc': 'p.confidence ASC, p.prediction_id DESC', 'confidence_desc': 'p.confidence DESC, p.prediction_id DESC',
    'ood_desc': 'p.ood_score DESC, p.prediction_id DESC', 'margin_asc': 'p.margin ASC, p.prediction_id DESC',
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


@dataclass(frozen=True)
class Filters:
    """One filter set shared by listing, counting, aggregates, the review queue and CSV export.
    Timestamps are ISO-8601 strings; *_from is inclusive, *_before exclusive (string comparison is correct for ISO)."""
    model_version: str
    label: str | None = None
    status: str | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    received_from: str | None = None
    received_before: str | None = None
    captured_from: str | None = None
    captured_before: str | None = None
    bbox: tuple[float, float, float, float] | None = None  # min_lon, min_lat, max_lon, max_lat
    flag: str | None = None                                # a quality flag name, or 'none' for tiles without flags
    reviewed: bool | None = None
    q: str | None = None                                   # tile-id prefix or part of the source file name

    def where(self) -> tuple[str, list]:
        clauses, args = ['p.model_version = ?'], [self.model_version]
        simple = (('p.label = ?', self.label), ('p.status = ?', self.status), ('p.confidence >= ?', self.min_confidence),
                  ('p.confidence <= ?', self.max_confidence), ('t.received_at >= ?', self.received_from),
                  ('t.received_at < ?', self.received_before), ('t.captured_at >= ?', self.captured_from),
                  ('t.captured_at < ?', self.captured_before))
        for clause, value in simple:
            if value is not None:
                clauses.append(clause); args.append(value)
        if self.bbox is not None:
            clauses.append('t.lon BETWEEN ? AND ? AND t.lat BETWEEN ? AND ?')
            args += [self.bbox[0], self.bbox[2], self.bbox[1], self.bbox[3]]
        if self.flag == 'none':
            clauses.append("t.quality_flags = '[]'")
        elif self.flag:
            clauses.append('t.quality_flags LIKE ?'); args.append(f'%"{self.flag}"%')
        if self.reviewed is not None:
            clauses.append('r.label IS NOT NULL' if self.reviewed else 'r.label IS NULL')
        if self.q:
            clauses.append('(t.tile_id LIKE ? OR t.source_name LIKE ?)'); args += [self.q.lower() + '%', f'%{self.q}%']
        return ' WHERE ' + ' AND '.join(clauses), args


class BlobStore:
    """blobs/<ab>/<cd>/<sha256>.png. Writes are atomic (temp file + rename) and idempotent."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, sha256: str, data: bytes) -> str:
        rel = f'{sha256[:2]}/{sha256[2:4]}/{sha256}.png'
        path = self.root / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f'.tmp{os.getpid()}.{threading.get_ident()}')
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return rel

    def path(self, rel: str) -> Path:
        return self.root / rel


class Repository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()  # SQLite allows one writer; serialise in-process writers explicitly
        with closing(self._connect()) as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    # ---------------------------------------------------------------- writes
    def register_model(self, manifest: dict) -> None:
        with self._write_lock, closing(self._connect()) as conn, conn:
            conn.execute('INSERT OR IGNORE INTO models VALUES (?, ?, ?, ?, ?)',
                         (manifest['model_version'], json.dumps(manifest['classes']), manifest['onnx']['sha256'],
                          json.dumps(manifest), utcnow()))

    def save(self, tile: dict, prediction: dict) -> int:
        """Tile row (if new) + prediction row in one transaction. Returns the prediction id."""
        with self._write_lock, closing(self._connect()) as conn, conn:
            conn.execute("""INSERT OR IGNORE INTO tiles (tile_id, source_name, received_at, captured_at, lat, lon, sensor,
                            width, height, bands, byte_size, quality_flags, blob_path)
                            VALUES (:tile_id, :source_name, :received_at, :captured_at, :lat, :lon, :sensor,
                            :width, :height, :bands, :byte_size, :quality_flags, :blob_path)""", tile)
            cur = conn.execute("""INSERT INTO predictions (tile_id, model_version, policy_version, status, label, confidence,
                                  margin, probabilities, ood_score, latency_ms, error, created_at)
                                  VALUES (:tile_id, :model_version, :policy_version, :status, :label, :confidence,
                                  :margin, :probabilities, :ood_score, :latency_ms, :error, :created_at)""", prediction)
            return cur.lastrowid

    def add_review(self, tile_id: str, label: str, note: str | None, reviewer: str | None) -> dict | None:
        """Append an analyst label. Returns None if the tile is unknown."""
        with self._write_lock, closing(self._connect()) as conn, conn:
            if conn.execute('SELECT 1 FROM tiles WHERE tile_id = ?', (tile_id,)).fetchone() is None:
                return None
            cur = conn.execute('INSERT INTO reviews (tile_id, label, note, reviewer, created_at) VALUES (?, ?, ?, ?, ?)',
                               (tile_id, label, note, reviewer, utcnow()))
            return dict(conn.execute('SELECT * FROM reviews WHERE review_id = ?', (cur.lastrowid,)).fetchone())

    # ---------------------------------------------------------------- level 1: lookup
    def current_result(self, tile_id: str, model_version: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(RESULT_SQL + ' WHERE p.tile_id = ? AND p.model_version = ?', (tile_id, model_version)).fetchone()
        return _decode(row) if row else None

    def tile_history(self, tile_id: str) -> dict | None:
        """The tile's metadata, every prediction ever made for it (all model versions, failures too) and every review."""
        with closing(self._connect()) as conn:
            tile = conn.execute('SELECT * FROM tiles WHERE tile_id = ?', (tile_id,)).fetchone()
            if tile is None:
                return None
            preds = conn.execute('SELECT * FROM predictions WHERE tile_id = ? ORDER BY prediction_id', (tile_id,)).fetchall()
            reviews = conn.execute('SELECT * FROM reviews WHERE tile_id = ? ORDER BY review_id', (tile_id,)).fetchall()
        t = dict(tile)
        t['quality_flags'] = json.loads(t['quality_flags'])
        return dict(tile=t, predictions=[_decode_prediction(dict(p)) for p in preds], reviews=[dict(r) for r in reviews])

    def blob_path(self, tile_id: str) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute('SELECT blob_path FROM tiles WHERE tile_id = ?', (tile_id,)).fetchone()
        return row['blob_path'] if row else None

    # ---------------------------------------------------------------- level 2: filter
    def list_results(self, filters: Filters | None = None, *, limit: int = 50, offset: int = 0, sort: str = 'newest',
                     **kwargs) -> list[dict]:
        f = filters or Filters(**kwargs)
        where, args = f.where()
        sql = RESULT_SQL + where + f' ORDER BY {SORTS[sort]} LIMIT ? OFFSET ?'
        with closing(self._connect()) as conn:
            return [_decode(r) for r in conn.execute(sql, (*args, limit, offset)).fetchall()]

    def count_results(self, filters: Filters) -> int:
        where, args = filters.where()
        with closing(self._connect()) as conn:
            return conn.execute('SELECT COUNT(*) ' + FROM_SQL + where, args).fetchone()[0]

    def iter_results(self, filters: Filters, sort: str = 'newest', chunk: int = 1000):
        """All matching rows, in chunks (for CSV export)."""
        offset = 0
        while True:
            rows = self.list_results(filters, limit=chunk, offset=offset, sort=sort)
            yield from rows
            if len(rows) < chunk:
                return
            offset += chunk

    # ---------------------------------------------------------------- level 3: aggregate
    def aggregates(self, filters: Filters) -> dict:
        where, args = filters.where()
        with closing(self._connect()) as conn:
            by_label = {r[0]: r[1] for r in conn.execute('SELECT p.label, COUNT(*) ' + FROM_SQL + where + ' AND p.label IS NOT NULL GROUP BY p.label', args)}
            by_status = {r[0]: r[1] for r in conn.execute('SELECT p.status, COUNT(*) ' + FROM_SQL + where + ' GROUP BY p.status', args)}
            by_day = [dict(day=r[0], label=r[1], n=r[2]) for r in conn.execute(
                'SELECT substr(t.received_at, 1, 10) AS day, p.label, COUNT(*) ' + FROM_SQL + where +
                ' AND p.label IS NOT NULL GROUP BY day, p.label ORDER BY day', args)]
            reviewed = conn.execute('SELECT COUNT(*), COALESCE(SUM(r.label = p.label), 0) ' + FROM_SQL + where + ' AND r.label IS NOT NULL', args).fetchone()
            review_labels = {r[0]: r[1] for r in conn.execute('SELECT r.label, COUNT(*) ' + FROM_SQL + where + ' AND r.label IS NOT NULL GROUP BY r.label', args)}
        return dict(by_label=by_label, by_status=by_status, by_day=by_day,
                    reviewed=dict(n=reviewed[0], agree=reviewed[1], by_label=review_labels))

    # ---------------------------------------------------------------- level 4: review queue
    def review_queue(self, filters: Filters, *, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        """Not-yet-reviewed tiles the policy did not accept: unfamiliar first (most unlike training data first), then
        needs_review (least decisive first)."""
        where, args = filters.where()
        where += " AND r.label IS NULL AND p.status IN ('unfamiliar', 'needs_review')"
        order = " ORDER BY CASE p.status WHEN 'unfamiliar' THEN 0 ELSE 1 END, CASE p.status WHEN 'unfamiliar' THEN -p.ood_score ELSE p.margin END, p.prediction_id"
        with closing(self._connect()) as conn:
            total = conn.execute('SELECT COUNT(*) ' + FROM_SQL + where, args).fetchone()[0]
            rows = conn.execute(RESULT_SQL + where + order + ' LIMIT ? OFFSET ?', (*args, limit, offset)).fetchall()
        return [_decode(r) for r in rows], total


def _percentile(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def ops_metrics(repo: 'Repository', model_version: str, since: str, recent: int = 500) -> dict:
    """What the health page plots: activity per day (every prediction row, any model: it is what the service did),
    recent latency and class mix (active model), storage and review backlog."""
    with closing(repo._connect()) as conn:
        q = lambda sql, *a: conn.execute(sql, a).fetchall()
        rows = q("SELECT substr(created_at, 1, 10), status, confidence, latency_ms FROM predictions WHERE created_at >= ? "
                 "ORDER BY prediction_id DESC LIMIT 200000", since)
        lat = [r[0] for r in q("SELECT latency_ms FROM predictions WHERE status != 'failed' AND latency_ms IS NOT NULL "
                                "ORDER BY prediction_id DESC LIMIT ?", recent)]
        mix = [r[0] for r in q("SELECT label FROM current_predictions WHERE model_version = ? AND label IS NOT NULL "
                                "ORDER BY prediction_id DESC LIMIT ?", model_version, recent)]
        current = dict(q("SELECT status, COUNT(*) FROM current_predictions WHERE model_version = ? GROUP BY status", model_version))
        totals = dict(tiles=q('SELECT COUNT(*) FROM tiles')[0][0], predictions=q('SELECT COUNT(*) FROM predictions')[0][0],
                      reviews=q('SELECT COUNT(*) FROM reviews')[0][0], blob_bytes=q('SELECT COALESCE(SUM(byte_size), 0) FROM tiles')[0][0])
        last = q('SELECT MAX(created_at) FROM predictions')[0][0]
        queue = q("SELECT COUNT(*) FROM current_predictions p LEFT JOIN current_reviews r USING (tile_id) "
                  "WHERE p.model_version = ? AND r.label IS NULL AND p.status IN ('unfamiliar', 'needs_review')", model_version)[0][0]
        agree = q("SELECT COUNT(*), COALESCE(SUM(r.label = p.label), 0) FROM current_predictions p JOIN current_reviews r USING (tile_id) "
                  "WHERE p.model_version = ?", model_version)[0]
    days = {}
    for day, status, conf, ms in rows:
        d = days.setdefault(day, dict(day=day, n=0, by_status={}, conf=[], ms=[]))
        d['n'] += 1
        d['by_status'][status] = d['by_status'].get(status, 0) + 1
        if conf is not None: d['conf'].append(conf)
        if ms is not None and status != 'failed': d['ms'].append(ms)
    daily = []
    for day in sorted(days):
        d = days[day]
        daily.append(dict(day=day, n=d['n'], by_status=d['by_status'],
                          mean_confidence=sum(d['conf']) / len(d['conf']) if d['conf'] else None,
                          unfamiliar_rate=d['by_status'].get('unfamiliar', 0) / d['n'],
                          p50_ms=_percentile(d['ms'], 50), p95_ms=_percentile(d['ms'], 95)))
    db_bytes = sum(os.path.getsize(p) for p in (repo.db_path, Path(f'{repo.db_path}-wal')) if Path(p).exists())
    return dict(daily=daily, latency_ms=dict(n=len(lat), p50=_percentile(lat, 50), p95=_percentile(lat, 95), max=max(lat) if lat else None),
                class_mix=dict(n=len(mix), counts={c: mix.count(c) for c in sorted(set(mix))}),
                current_by_status=current, totals=dict(totals, db_bytes=db_bytes), last_prediction_at=last,
                review_queue=queue, reviews=dict(n=agree[0], agree=agree[1]))


def _decode_prediction(d: dict) -> dict:
    if d.get('probabilities'):
        d['probabilities'] = json.loads(d['probabilities'])
    return d


def _decode(row: sqlite3.Row) -> dict:
    d = _decode_prediction(dict(row))
    d['quality_flags'] = json.loads(d['quality_flags'])
    return d

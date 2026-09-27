"""HTTP API. Run with:  uvicorn tile_service.api:app --host 127.0.0.1 --port 8000

Core path (Part 2):
  POST /v1/tiles                   upload one tile -> classify -> store -> result      (201 new, 200 idempotent hit)
Querying the results (DESIGN_NOTE.md §4.3, levels 1-4 + export):
  GET  /v1/tiles/{tile_id}         1 lookup: the tile, every prediction made for it (all model versions) and its reviews
  GET  /v1/tiles/{tile_id}/image   the stored raw tile (PNG)
  GET  /v1/predictions             2 filter: label / status / confidence / dates / bbox / flag / reviewed / search, sorted,
                                     paged (total in the X-Total-Count header)
  GET  /v1/predictions.csv         the same filters, every matching row, as CSV
  GET  /v1/stats                   3 aggregate: per-class counts, raw and corrected with the eval confusion matrix (+95% band)
  GET  /v1/review-queue            4 review queue: unreviewed tiles the policy did not accept, most useful first
  POST /v1/tiles/{tile_id}/label   store an analyst label (append-only ground truth)
  GET  /v1/meta                    classes, statuses, flags, policy thresholds, calibration status (drives the UI)
Operations and UI:
  GET  /healthz                    model checksum + self-test, database, disk (machine-readable; for monitors)
  GET  /v1/ops                     what the /health page draws: checks, alerts, per-day activity, latency, class mix
  GET  /health                     the visual health dashboard
  GET  /  and  /explore            self-contained browser pages: classify a tile / query the stored results
  GET  /docs                       Swagger UI served from tile_service/static (no CDN: works on an air-gapped box)
"""
import csv
import io
import logging
import shutil
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import auth
from .aggregate import corrected_shares, load_calibration
from .config import Settings
from .model import OnnxTileModel
from .pipeline import TileMetadata, TilePipeline
from .schemas import Health, IngestResult, QueueItem, ReviewIn, TileResult
from .storage import SORTS, BlobStore, Filters, Repository, ops_metrics
from .validation import InvalidTile

log = logging.getLogger('tile_service')
STATUSES = ('confident', 'needs_review', 'unfamiliar', 'failed')
QUALITY_FLAGS = ('blank', 'grey_low_texture', 'saturated', 'nodata_pixels')
EXTRA_REVIEW_LABELS = ('Other', 'Unusable')  # "none of the classes" and "cloud / no-data / not assessable"
STATIC_DIR = Path(__file__).resolve().parent / 'static'

# health thresholds (shown on the /health page next to the values they judge)
DISK_WARN_MB, DISK_CRITICAL_MB = 1000, 100
LATENCY_WARN_MS, LATENCY_SERIOUS_MS = 100, 250           # p95 model time per tile
UNFAMILIAR_WARN, UNFAMILIAR_SERIOUS = 0.05, 0.15          # share of recent tiles; ~1% expected by construction
MIN_TILES_FOR_RATES = 50                                  # below this, a rate is noise and is not judged


def _utc_iso(value: str, field: str) -> str:
    """ISO date or datetime -> normalised UTC ISO string (naive = UTC), so string comparison in SQL is correct."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(400, detail=f'{field} must be an ISO-8601 date or timestamp') from None
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.isoformat(timespec='milliseconds')


def _range(frm: str | None, to: str | None, field: str) -> tuple[str | None, str | None]:
    """A `to` given as a plain date is inclusive of that whole day."""
    lo = _utc_iso(frm, f'{field}_from') if frm else None
    hi = None
    if to:
        try:
            hi = _utc_iso((date.fromisoformat(to) + timedelta(days=1)).isoformat(), f'{field}_to')
        except ValueError:
            hi = _utc_iso(to, f'{field}_to')
    return lo, hi


def create_app(settings: Settings | None = None, model=None) -> FastAPI:
    """`model` lets tests inject a fake; by default the ONNX artifact in settings.model_dir is loaded at startup."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
        app.state.repo = Repository(settings.db_path)
        app.state.blobs = BlobStore(settings.blob_dir)
        app.state.model, app.state.model_error, app.state.self_test = None, None, None
        try:
            m = model or OnnxTileModel(settings.model_dir, threads=settings.ort_threads)
            app.state.self_test = m.self_test()
            if not app.state.self_test['passed']:
                raise RuntimeError(f"self-test failed: {app.state.self_test['tiles']}")
            app.state.repo.register_model(m.manifest)
            app.state.model = m
            app.state.pipeline = TilePipeline(m, app.state.repo, app.state.blobs, settings.max_upload_bytes)
            log.info('serving model %s from %s', m.version, settings.model_dir)
        except Exception as e:  # keep the process up so /healthz can say what is wrong; ingest returns 503
            app.state.model_error = f'{type(e).__name__}: {e}'
            log.error('model not loaded: %s', app.state.model_error)
        yield

    # FastAPI's default /docs page loads Swagger UI from a CDN and ReDoc from another; both fail offline.
    # Serve Swagger UI from vendored files instead (fetched once by scripts/fetch_offline_assets.py) and drop ReDoc.
    app = FastAPI(title='Offline land-use tile classifier', version='0.2.0', lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
    if settings.access_key:  # for exposure through a shareable link; see auth.py
        auth.install(app, settings.access_key)

    # the pages change with the code; no-cache = the browser revalidates every time instead of reusing a stale copy
    page = lambda name: FileResponse(STATIC_DIR / 'ui' / name, headers={'Cache-Control': 'no-cache'})

    @app.get('/', include_in_schema=False)
    def classify_page():
        return page('index.html')

    @app.get('/health', include_in_schema=False)
    def health_page():
        return page('health.html')

    @app.get('/explore', include_in_schema=False)
    def explore_page():
        return page('explore.html')

    @app.get('/docs', include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url=app.openapi_url, title=f'{app.title} - docs',
                                   swagger_js_url='/static/swagger-ui/swagger-ui-bundle.js',
                                   swagger_css_url='/static/swagger-ui/swagger-ui.css',
                                   swagger_favicon_url='/static/swagger-ui/favicon-32x32.png',
                                   swagger_ui_parameters={'validatorUrl': None})  # no calls to validator.swagger.io

    def active_model():
        if app.state.model is None:
            raise HTTPException(503, detail=f'model unavailable: {app.state.model_error}')
        return app.state.model

    def review_labels() -> list[str]:
        return list(active_model().classes) + list(EXTRA_REVIEW_LABELS)

    def filters(label: str | None = None,
                status: str | None = Query(None, pattern='^(' + '|'.join(STATUSES) + ')$'),
                min_confidence: float | None = Query(None, ge=0, le=1),
                max_confidence: float | None = Query(None, ge=0, le=1),
                received_from: str | None = Query(None, description='ISO date/timestamp, inclusive'),
                received_to: str | None = Query(None, description='ISO date (whole day inclusive) or timestamp'),
                captured_from: str | None = None, captured_to: str | None = None,
                bbox: str | None = Query(None, description='min_lon,min_lat,max_lon,max_lat (tiles without coordinates never match)'),
                flag: str | None = Query(None, pattern='^(' + '|'.join(QUALITY_FLAGS + ('none',)) + ')$'),
                reviewed: bool | None = None,
                q: str | None = Query(None, max_length=100, description='tile-id prefix or part of the file name'),
                model_version: str | None = Query(None, description='default: the active model')) -> Filters:
        label = label or None  # an empty form field means "no filter"
        if label is not None and label not in active_model().classes:
            raise HTTPException(400, detail=f'unknown label {label!r}; expected one of {active_model().classes}')
        box = None
        if bbox:
            try:
                box = tuple(float(v) for v in bbox.split(','))
                assert len(box) == 4 and -180 <= box[0] <= box[2] <= 180 and -90 <= box[1] <= box[3] <= 90
            except (ValueError, AssertionError):
                raise HTTPException(400, detail='bbox must be min_lon,min_lat,max_lon,max_lat') from None
        rf, rt = _range(received_from, received_to, 'received')
        cf, ct = _range(captured_from, captured_to, 'captured')
        return Filters(model_version=model_version or active_model().version, label=label, status=status,
                       min_confidence=min_confidence, max_confidence=max_confidence, received_from=rf, received_before=rt,
                       captured_from=cf, captured_before=ct, bbox=box, flag=flag, reviewed=reviewed, q=q or None)

    # ------------------------------------------------------------------ core path
    @app.post('/v1/tiles', response_model=IngestResult, status_code=201,
              responses={200: {'description': 'already classified by this model version (idempotent hit)'},
                         400: {'description': 'not a valid 64x64 RGB PNG tile'}, 500: {'description': 'stored, but inference failed'},
                         503: {'description': 'model not loaded'}})
    def ingest_tile(response: Response,
                    file: UploadFile = File(..., description='64x64 RGB PNG tile'),
                    captured_at: str | None = Form(None, description='ISO-8601 capture time, if known (stored as UTC)'),
                    lat: float | None = Form(None, ge=-90, le=90),
                    lon: float | None = Form(None, ge=-180, le=180),
                    sensor: str | None = Form(None, max_length=64)):
        active_model()
        captured = _utc_iso(captured_at, 'captured_at') if captured_at is not None else None
        data = file.file.read(settings.max_upload_bytes + 1)  # read one byte past the limit so oversize is detectable
        try:
            result = app.state.pipeline.process(data, TileMetadata(file.filename, captured, lat, lon, sensor))
        except InvalidTile as e:
            raise HTTPException(400, detail=str(e)) from None
        body = IngestResult(**result.record, idempotent_hit=not result.created)
        if body.status == 'failed':
            return JSONResponse(status_code=500, content=body.model_dump())
        if not result.created:
            response.status_code = 200
        return body

    # ------------------------------------------------------------------ 1. lookup
    @app.get('/v1/tiles/{tile_id}')
    def get_tile(tile_id: str):
        found = app.state.repo.tile_history(tile_id)
        if found is None:
            raise HTTPException(404, detail='unknown tile_id')
        return found

    @app.get('/v1/tiles/{tile_id}/image', response_class=FileResponse)
    def get_tile_image(tile_id: str):
        rel = app.state.repo.blob_path(tile_id)
        if rel is None:
            raise HTTPException(404, detail='unknown tile_id')
        path = app.state.blobs.path(rel).resolve()
        if app.state.blobs.root.resolve() not in path.parents or not path.exists():
            raise HTTPException(404, detail='tile image missing')
        # content-addressed, so it never changes
        return FileResponse(path, media_type='image/png', headers={'Cache-Control': 'private, max-age=86400, immutable'})

    # ------------------------------------------------------------------ 2. filter (+ export)
    @app.get('/v1/predictions', response_model=list[TileResult])
    def list_predictions(response: Response, f: Filters = Depends(filters),
                         sort: str = Query('newest', pattern='^(' + '|'.join(SORTS) + ')$'),
                         limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        """Current predictions (latest per tile) of the active model, filtered. Total match count in X-Total-Count."""
        response.headers['X-Total-Count'] = str(app.state.repo.count_results(f))
        return app.state.repo.list_results(f, limit=limit, offset=offset, sort=sort)

    @app.get('/v1/predictions.csv', response_class=StreamingResponse)
    def export_predictions(f: Filters = Depends(filters), sort: str = Query('newest', pattern='^(' + '|'.join(SORTS) + ')$')):
        classes = active_model().classes
        cols = ['tile_id', 'source_name', 'received_at', 'captured_at', 'lat', 'lon', 'sensor', 'model_version', 'policy_version',
                'status', 'label', 'confidence', 'margin', 'ood_score', 'quality_flags'] + [f'p_{c}' for c in classes] + \
               ['review_label', 'reviewer', 'reviewed_at']

        def rows():
            buf = io.StringIO(); w = csv.writer(buf)
            w.writerow(cols); yield buf.getvalue()
            for r in app.state.repo.iter_results(f, sort=sort):
                buf.seek(0); buf.truncate()
                probs = r.get('probabilities') or {}
                w.writerow([r.get(c) for c in cols[:14]] + [';'.join(r['quality_flags'])] + [probs.get(c) for c in classes] +
                           [r.get('review_label'), r.get('reviewer'), r.get('reviewed_at')])
                yield buf.getvalue()
        name = f"predictions-{f.model_version}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.csv"
        return StreamingResponse(rows(), media_type='text/csv', headers={'Content-Disposition': f'attachment; filename="{name}"'})

    # ------------------------------------------------------------------ 3. aggregate
    @app.get('/v1/stats')
    def stats(f: Filters = Depends(filters)):
        """Per-class counts of the filtered tiles: raw (predicted labels) and corrected for the model's measured confusions."""
        m = active_model()
        agg = app.state.repo.aggregates(f)
        counts = np.array([agg['by_label'].get(c, 0) for c in m.classes], dtype=float)
        n = int(counts.sum())
        cal = load_calibration(getattr(m, 'dir', None) or settings.model_dir, m.version, m.classes)
        corr = corrected_shares(counts, cal['confusion_counts']) if cal and n else None
        by_class = []
        for i, c in enumerate(m.classes):
            row = dict(label=c, raw_count=int(counts[i]), raw_share=counts[i] / n if n else 0.0)
            if corr:
                row.update(corrected_share=corr['share'][i], corrected_low=corr['low'][i], corrected_high=corr['high'][i],
                           corrected_count=corr['share'][i] * n)
            by_class.append(row)
        days = {}
        for r in agg['by_day']:
            days.setdefault(r['day'], {})[r['label']] = r['n']
        rv = agg['reviewed']
        return dict(model_version=m.version, n_classified=n, n_failed=agg['by_status'].get('failed', 0),
                    by_status={s: agg['by_status'].get(s, 0) for s in STATUSES}, classes=m.classes, by_class=by_class,
                    correction=dict(available=bool(corr),
                                    method='adjusted classify-and-count: solve p_pred = M^T p_true with M = P(predicted | true); '
                                           '95% band from a bootstrap over the calibration matrix and the observed labels',
                                    source=cal.get('source') if cal else None, n_labelled=cal.get('n') if cal else None,
                                    reason=None if corr else ('no tiles match' if not n else 'no calibration file for this model '
                                                              '(run training/evaluate.py)')),
                    by_day=[dict(day=d, counts=v) for d, v in sorted(days.items())],
                    reviews=dict(n=rv['n'], agree=rv['agree'], agreement=rv['agree'] / rv['n'] if rv['n'] else None,
                                 by_label=rv['by_label']))

    # ------------------------------------------------------------------ 4. review queue
    def reason(r: dict, policy: dict) -> str:
        if r['status'] == 'unfamiliar':
            return f"unlike the training data: OOD score {r['ood_score']:.3f} > threshold {policy['ood_threshold']:.3f}"
        flags = [fl for fl in r['quality_flags'] if fl in ('blank', 'grey_low_texture')]
        if flags:
            return f"quality flag {', '.join(flags)}: the image may be haze, cloud or no-data rather than land"
        return f"low confidence {r['confidence']:.2f} < {policy['min_confidence']:.2f} (top-2 margin {r['margin']:.2f})"

    @app.get('/v1/review-queue', response_model=list[QueueItem])
    def review_queue(response: Response, f: Filters = Depends(filters),
                     limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        """Unreviewed tiles the policy did not accept: `unfamiliar` first (most unlike training first), then
        `needs_review` (least decisive first). Labelling a tile removes it. Total in X-Total-Count."""
        policy = active_model().policy
        rows, total = app.state.repo.review_queue(f, limit=limit, offset=offset)
        response.headers['X-Total-Count'] = str(total)
        return [QueueItem(**r, reason=reason(r, policy)) for r in rows]

    @app.post('/v1/tiles/{tile_id}/label', status_code=201)
    def label_tile(tile_id: str, body: ReviewIn):
        allowed = review_labels()
        if body.label not in allowed:
            raise HTTPException(400, detail=f'label must be one of {allowed}')
        review = app.state.repo.add_review(tile_id, body.label, body.note, body.reviewer)
        if review is None:
            raise HTTPException(404, detail='unknown tile_id')
        return review

    @app.get('/v1/meta')
    def meta():
        m = active_model()
        cal = load_calibration(getattr(m, 'dir', None) or settings.model_dir, m.version, m.classes)
        return dict(model_version=m.version, classes=m.classes, review_labels=review_labels(), statuses=STATUSES,
                    quality_flags=QUALITY_FLAGS, sorts=list(SORTS), policy=m.policy,
                    calibration=dict(available=bool(cal), source=cal.get('source') if cal else None, n=cal.get('n') if cal else None))

    # ------------------------------------------------------------------ operations
    started = time.time()

    @app.get('/v1/ops')
    def ops(days: int = Query(14, ge=1, le=90)):
        """Everything the health dashboard shows. Checks use the status scale good / warning / serious / critical."""
        m = app.state.model
        now = datetime.now(timezone.utc)
        since = (now - timedelta(days=days - 1)).date().isoformat()
        met = ops_metrics(app.state.repo, getattr(m, 'version', ''), since)
        disk = shutil.disk_usage(settings.data_dir)
        free_mb = disk.free / 1e6
        checks = []

        def check(name, status, detail):
            checks.append(dict(name=name, status=status, detail=detail))

        check('Model artifact', 'good' if m else 'critical',
              f'{m.version}: files match their SHA-256 checksums' if m else f'not loaded: {app.state.model_error}')
        st = app.state.self_test
        if st:
            worst = max((t['max_prob_diff'] for t in st['tiles']), default=0.0)
            check('Self-test', 'good' if st['passed'] else 'critical',
                  f"{sum(t['ok'] for t in st['tiles'])}/{len(st['tiles'])} golden tiles reproduced (worst |Δp| {worst:.1e})")
        else:
            check('Self-test', 'critical', 'did not run: model not loaded')
        try:
            app.state.repo.current_result('healthcheck', 'healthcheck'); db_ok = True
        except Exception:
            db_ok = False
        check('Database', 'good' if db_ok else 'critical', 'SQLite readable and writable' if db_ok else 'SQLite query failed')
        check('Disk space', 'critical' if free_mb < DISK_CRITICAL_MB else 'warning' if free_mb < DISK_WARN_MB else 'good',
              f'{free_mb / 1e3:.1f} GB free of {disk.total / 1e9:.0f} GB (warning below {DISK_WARN_MB / 1e3:.0f} GB)')
        week = met['daily'][-7:]
        n7 = sum(d['n'] for d in week)
        failed7 = sum(d['by_status'].get('failed', 0) for d in week)
        check('Inference failures (7 days)', 'good' if not failed7 else 'warning',
              f'{failed7} failed of {n7} attempts' if n7 else 'no activity in the last 7 days')
        unf7 = sum(d['by_status'].get('unfamiliar', 0) for d in week)
        rate = unf7 / n7 if n7 else None
        if n7 >= MIN_TILES_FOR_RATES:
            check('Unfamiliar-input rate (7 days)', 'serious' if rate > UNFAMILIAR_SERIOUS else 'warning' if rate > UNFAMILIAR_WARN else 'good',
                  f'{rate:.1%} of {n7} tiles (≈1% expected; warning above {UNFAMILIAR_WARN:.0%}): a rise means inputs are drifting '
                  'away from the training data')
        else:
            check('Unfamiliar-input rate (7 days)', 'good', f'only {n7} tiles in the last 7 days: too few to judge')
        p95 = met['latency_ms']['p95']
        if p95 is not None:
            check('Latency (p95, last 500)', 'serious' if p95 > LATENCY_SERIOUS_MS else 'warning' if p95 > LATENCY_WARN_MS else 'good',
                  f'{p95:.0f} ms per tile (budget {LATENCY_WARN_MS} ms)')
        rank = {'good': 0, 'warning': 1, 'serious': 2, 'critical': 3}
        worst_status = max((c['status'] for c in checks), key=rank.get)
        return dict(now=now.isoformat(timespec='seconds'), started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(timespec='seconds'),
                    uptime_s=time.time() - started, status=worst_status, checks=checks,
                    model=dict(version=getattr(m, 'version', None), classes=getattr(m, 'classes', None), policy=getattr(m, 'policy', None), self_test=st,
                               expected_unfamiliar_rate=(1 - m.policy.get('ood_percentile', 99) / 100) if m else None),
                    thresholds=dict(unfamiliar_warn=UNFAMILIAR_WARN, unfamiliar_serious=UNFAMILIAR_SERIOUS, latency_warn_ms=LATENCY_WARN_MS,
                                    latency_serious_ms=LATENCY_SERIOUS_MS, disk_warn_mb=DISK_WARN_MB, min_tiles_for_rates=MIN_TILES_FOR_RATES),
                    disk=dict(free_bytes=disk.free, total_bytes=disk.total), window_days=days, since=since, **met)

    @app.get('/v1/ops/selftest/{name}', include_in_schema=False)
    def selftest_image(name: str):
        """The golden self-test tiles shipped inside the model artifact (only names listed in its manifest are served)."""
        m = active_model()
        files = {Path(t['file']).name: t['file'] for t in m.manifest.get('selftest', {}).get('tiles', [])}
        if name not in files or not getattr(m, 'dir', None):
            raise HTTPException(404, detail='unknown self-test tile')
        return FileResponse(Path(m.dir) / files[name], media_type='image/png')

    @app.get('/healthz', response_model=Health)
    def health(response: Response):
        try:
            app.state.repo.current_result('healthcheck', 'healthcheck')
            db_ok = True
        except Exception:
            db_ok = False
        free_mb = shutil.disk_usage(settings.data_dir).free / 1e6
        ok = app.state.model is not None and db_ok and free_mb > 100
        if not ok:
            response.status_code = 503
        return Health(status='ok' if ok else 'degraded', model_version=getattr(app.state.model, 'version', None),
                      model_error=app.state.model_error, self_test=app.state.self_test, database_ok=db_ok, disk_free_mb=round(free_mb, 1))

    return app


app = create_app()

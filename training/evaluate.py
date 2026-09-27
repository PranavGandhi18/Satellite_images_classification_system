"""Evaluate the trained model *through the service*: every eval tile goes through POST /v1/tiles, exactly like a client.

Also checks idempotency, rejection of invalid uploads, and how synthetic "never seen anything like this" tiles
(uniform grey, cloud-like, noise, checkerboard) are handled. eval_set is used here and nowhere else.

usage:
  python training/evaluate.py                          # in-process app (models/active, data dir from TILE_DATA_DIR or ./data)
  python training/evaluate.py --url http://127.0.0.1:8000   # against a running server
Writes reports/eval_results.json and reports/eval_report.md.
"""
import argparse
import csv
import io
import json
import math
import random
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATASET = ROOT / 'be-mlsys-assignment-dataset'
CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']


def png(arr, fmt='PNG'):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format=fmt); return buf.getvalue()


def synthetic_tiles():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:64, :64]
    checker = (((yy // 8) + (xx // 8)) % 2 * 255).astype(np.uint8)
    return {
        'uniform grey (no-data-like)': np.full((64, 64, 3), 120, np.uint8),
        'bright hazy white (cloud-like)': np.clip(225 + rng.normal(0, 4, (64, 64, 3)), 0, 255).astype(np.uint8),
        'random noise': rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
        'black/white checkerboard': np.stack([checker] * 3, -1),
        'saturated magenta': np.stack([np.full((64, 64), 255), np.zeros((64, 64)), np.full((64, 64), 255)], -1).astype(np.uint8),
    }


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', help='base URL of a running service; default: run the app in-process')
    ap.add_argument('--key', help='access key, if the service was started with TILE_ACCESS_KEY / TILE_ACCESS_KEY_FILE')
    ap.add_argument('--reports-dir', type=Path, default=ROOT / 'reports')
    args = ap.parse_args()

    headers = {'X-API-Key': args.key} if args.key else {}
    with ExitStack() as stack:
        if args.url:
            import httpx
            client, mode = stack.enter_context(httpx.Client(base_url=args.url, timeout=30, headers=headers)), f'HTTP against {args.url}'
        else:
            from fastapi.testclient import TestClient
            from tile_service.api import create_app
            client, mode = stack.enter_context(TestClient(create_app(), headers=headers)), 'in-process FastAPI app (same code path as the server)'
        out = evaluate(client, mode)
    args.reports_dir.mkdir(exist_ok=True)
    (args.reports_dir / 'eval_results.json').write_text(json.dumps(out, indent=2))
    write_calibration(out)
    (args.reports_dir / 'eval_report.md').write_text(render(out))
    print(render(out))


def write_calibration(out):
    """The eval confusion matrix, next to the model it was measured for. GET /v1/stats uses it to correct aggregate class
    counts for the model's known confusions. It is evaluation output, not part of the checksummed model artifact."""
    model_dir = ROOT / 'models' / out['model_version']
    if not model_dir.is_dir():
        print(f'(no local models/{out["model_version"]}/: calibration file not written)')
        return
    cal = dict(model_version=out['model_version'], classes=CLASSES, confusion_counts=out['confusion_matrix'], n=out['n'],
               source=f"eval_set: {out['n']} labelled tiles, never used for training or threshold choices",
               created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), rows='true class', columns='predicted class')
    (model_dir / 'aggregate_calibration.json').write_text(json.dumps(cal, indent=2))
    print(f'wrote models/{out["model_version"]}/aggregate_calibration.json')


def evaluate(client, mode):
    post = lambda name, data: client.post('/v1/tiles', files={'file': (name, data, 'image/png')})
    health = client.get('/healthz').json()
    assert health['status'] == 'ok', f'service not healthy: {health}'

    rows = list(csv.DictReader(open(DATASET / 'eval_labels.csv')))
    random.Random(0).shuffle(rows)  # eval filenames are sorted by class; never replay them in that order
    results, rtts = [], []
    for row in rows:
        t0 = time.perf_counter()
        r = post(row['filename'], (DATASET / 'eval_set' / row['filename']).read_bytes())
        rtts.append((time.perf_counter() - t0) * 1000)
        assert r.status_code in (200, 201), (row, r.status_code, r.text)
        b = r.json()
        results.append(dict(file=row['filename'], true=row['true_label'], pred=b['label'], status=b['status'], confidence=b['confidence'],
                            margin=b['margin'], ood_score=b['ood_score'], flags=b['quality_flags'], latency_ms=b['latency_ms'],
                            tile_id=b['tile_id'], idempotent_hit=b['idempotent_hit']))

    y = np.array([CLASSES.index(r['true']) for r in results]); p = np.array([CLASSES.index(r['pred']) for r in results])
    correct = y == p
    n, k = len(y), int(correct.sum())
    cm = np.zeros((7, 7), int)
    for a, b in zip(y, p): cm[a, b] += 1
    by_status = {}
    for s in ('confident', 'needs_review', 'unfamiliar'):
        m = np.array([r['status'] == s for r in results])
        by_status[s] = dict(n=int(m.sum()), share=float(m.mean()), accuracy=float(correct[m].mean()) if m.any() else None)
    lat = [r['latency_ms'] for r in results if not r['idempotent_hit']]  # empty if every tile was already stored

    # idempotency: the same bytes again must return the stored result, not a new prediction
    again = post(rows[0]['filename'], (DATASET / 'eval_set' / rows[0]['filename']).read_bytes())
    idem = dict(status_code=again.status_code, idempotent_hit=again.json()['idempotent_hit'],
                same_result=again.json()['created_at'] == client.get(f"/v1/tiles/{results[0]['tile_id']}").json()['predictions'][-1]['created_at'])

    rng = np.random.default_rng(1)
    invalid = {
        'garbage bytes': b'this is not an image',
        '32x32 PNG': png(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)),
        'JPEG instead of PNG': png(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), 'JPEG'),
        'grayscale PNG': png(rng.integers(0, 256, (64, 64), dtype=np.uint8)),
        'truncated PNG': png(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))[:300],
    }
    invalid_res = {name: dict(status_code=(r := post(name, data)).status_code, detail=r.json().get('detail')) for name, data in invalid.items()}

    synth = {}
    for name, arr in synthetic_tiles().items():
        b = post(name + '.png', png(arr)).json()
        synth[name] = dict(status=b['status'], label=b['label'], confidence=b['confidence'], ood_score=b['ood_score'], flags=b['quality_flags'])
    grey_real = [r for r in results if r['flags']]

    return dict(mode=mode, model_version=health['model_version'], n=n, accuracy=k / n, accuracy_ci95=wilson(k, n),
               confident_only=dict(coverage=by_status['confident']['share'], accuracy=by_status['confident']['accuracy']),
               by_status=by_status, per_class_recall={c: float(correct[y == i].mean()) for i, c in enumerate(CLASSES)},
               confusion_matrix=cm.tolist(), latency_ms=dict(model_p50=float(np.percentile(lat, 50)) if lat else None, model_p95=float(np.percentile(lat, 95)) if lat else None,
                                                             request_p50=float(np.percentile(rtts, 50)), request_p95=float(np.percentile(rtts, 95))),
               idempotency=idem, invalid_inputs=invalid_res, synthetic_tiles=synth,
               flagged_real_tiles=[{k2: r[k2] for k2 in ('file', 'true', 'pred', 'status', 'confidence', 'ood_score', 'flags')} for r in grey_real],
                errors=[{k2: r[k2] for k2 in ('file', 'true', 'pred', 'status', 'confidence', 'margin', 'ood_score')} for r in results if r['true'] != r['pred']])


def render(o):
    f = lambda x: '–' if x is None else f'{100 * x:.1f}%'
    ms = lambda l: ('– (every tile was already stored, so no inference ran)' if l['model_p50'] is None
                    else f"{l['model_p50']:.1f} / {l['model_p95']:.1f} ms")
    lines = [f"# Evaluation report\n", f"- Model: `{o['model_version']}`", f"- Mode: {o['mode']}",
             f"- Eval tiles: {o['n']} (sent in shuffled order; eval_set was not used for any training or threshold choice)\n",
             '## Accuracy\n', '| Measure | Value |', '|---|---|',
             f"| Accuracy, all tiles (label regardless of status) | **{f(o['accuracy'])}** (95% CI {f(o['accuracy_ci95'][0])}–{f(o['accuracy_ci95'][1])}) |",
             f"| Tiles marked `confident` (coverage) | {f(o['confident_only']['coverage'])} |",
             f"| Accuracy on `confident` tiles | **{f(o['confident_only']['accuracy'])}** |",
             f"| Model latency per tile, p50 / p95 | {ms(o['latency_ms'])} |",
             f"| Full request round trip, p50 / p95 | {o['latency_ms']['request_p50']:.1f} / {o['latency_ms']['request_p95']:.1f} ms |\n",
             '| Status | Tiles | Share | Accuracy |', '|---|---|---|---|']
    lines += [f"| {s} | {v['n']} | {f(v['share'])} | {f(v['accuracy'])} |" for s, v in o['by_status'].items()]
    lines += ['\n**Per-class recall:** ' + ', '.join(f'{c} {f(v)}' for c, v in o['per_class_recall'].items()) + '\n',
              '**Confusion matrix** (rows = true, columns = predicted):\n', '| | ' + ' | '.join(CLASSES) + ' |', '|---' * 8 + '|']
    lines += [f'| **{c}** | ' + ' | '.join(str(v) for v in row) + ' |' for c, row in zip(CLASSES, o['confusion_matrix'])]
    lines += ['\n**Misclassified tiles:**\n', '| File | True | Predicted | Status | Confidence | OOD score |', '|---|---|---|---|---|---|']
    lines += [f"| {e['file']} | {e['true']} | {e['pred']} | {e['status']} | {e['confidence']:.3f} | {e['ood_score']:.3f} |" for e in o['errors']] or ['| (none) | | | | | |']
    i = o['idempotency']
    lines += ['\n## Behaviour checks\n', f"- **Idempotency:** re-posting an already-classified tile returned HTTP {i['status_code']}, "
              f"`idempotent_hit={i['idempotent_hit']}`, same stored result: {i['same_result']}.",
              '- **Invalid uploads** (all must be rejected with 400 and a reason):\n', '| Upload | HTTP | Reason |', '|---|---|---|']
    lines += [f"| {k} | {v['status_code']} | {v['detail']} |" for k, v in o['invalid_inputs'].items()]
    lines += ['\n- **Synthetic tiles unlike anything in training:**\n', '| Tile | Status | Label | Confidence | OOD score | Quality flags |', '|---|---|---|---|---|---|']
    lines += [f"| {k} | {v['status']} | {v['label']} | {v['confidence']:.3f} | {v['ood_score']:.3f} | {', '.join(v['flags']) or '–'} |" for k, v in o['synthetic_tiles'].items()]
    lines += ['\n- **Real eval tiles that tripped a quality flag:**\n', '| File | True | Predicted | Status | Confidence | OOD score | Flags |', '|---|---|---|---|---|---|---|']
    lines += [f"| {r['file']} | {r['true']} | {r['pred']} | {r['status']} | {r['confidence']:.3f} | {r['ood_score']:.3f} | {', '.join(r['flags'])} |" for r in o['flagged_real_tiles']] or ['| (none) | | | | | | |']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()

"""Prove that the service - and retraining - run with no network access. Writes reports/offline_check.md.

Every Python process started here loads scripts/netguard/sitecustomize.py, which blocks and logs any DNS lookup or
connection that is not to localhost. Hugging Face / torch caches point at empty directories, so nothing can be served
from a download cache either. Steps:
  0. guard self-test          an outbound HTTPS request must be blocked and logged
  1. offline install          fresh venv, `pip install --no-index` from wheelhouse/serve only (no PyTorch in it)
  2. offline service          uvicorn from that venv: healthz, one eval tile per class, idempotent re-post, reads, /docs
  3. socket inspection        while it runs, every socket the server owns must be on 127.0.0.1
  4. clean guard log          the server logged zero blocked network attempts
  5. offline retraining       training/train.py --epochs 1 with vendored weights, same guard, empty caches
usage: python scripts/offline_check.py [--skip-train]
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / 'scripts' / 'netguard'
DATASET = ROOT / 'be-mlsys-assignment-dataset'
PORT = 8765


def guarded_env(tmp: Path, log: Path, **extra):
    # no proxy variables at all: any network attempt must go direct, so the guard sees (and logs) it
    env = {k: v for k, v in os.environ.items() if k not in ('VIRTUAL_ENV', 'PYTHONHOME') and 'proxy' not in k.lower()}
    env.update(PYTHONPATH=f'{GUARD}{os.pathsep}{ROOT}', NETGUARD_LOG=str(log),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(tmp / 'empty-hf-cache'),
               TORCH_HOME=str(tmp / 'empty-torch-cache'), XDG_CACHE_HOME=str(tmp / 'empty-cache'),
               PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    env.update(extra)
    return env


def events(log: Path):
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def blocked(log: Path):
    return [e for e in events(log) if e['event'].startswith('blocked')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='offline-check-'))
    checks = []

    def check(name, ok, detail=''):
        checks.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)

    base_python = Path(subprocess.check_output([sys.executable, '-c', 'import sys; print(sys.base_prefix)'], text=True).strip()) / 'bin' / 'python3'

    # 0. the guard itself works
    log0 = tmp / 'guard-selftest.log'
    code = ("import socket, urllib.request\n"
            "for f in (lambda: urllib.request.urlopen('https://huggingface.co', timeout=5), lambda: socket.socket().connect(('1.1.1.1', 443))):\n"
            "    try:\n        f(); print('REACHED')\n    except Exception as e:\n        print('BLOCKED', type(e).__name__)")
    out = subprocess.run([str(base_python), '-c', code], env=guarded_env(tmp, log0), capture_output=True, text=True).stdout.split()
    got = [e['event'] for e in blocked(log0)]
    check('guard blocks and logs outbound traffic (DNS lookup + raw IP connect)', 'REACHED' not in out and got == ['blocked_dns', 'blocked_connect'],
          f'logged: {got}')

    # 1. fresh venv, serving dependencies only, installed from the wheelhouse with no index
    venv, log1 = tmp / 'serve-venv', tmp / 'install.log'
    subprocess.run([str(base_python), '-m', 'venv', str(venv)], check=True, env=guarded_env(tmp, log1))
    pip = subprocess.run([str(venv / 'bin' / 'python'), '-m', 'pip', 'install', '-q', '--no-index', '--find-links',
                          str(ROOT / 'wheelhouse' / 'serve'), '-r', str(ROOT / 'requirements-serve.txt')],
                         env=guarded_env(tmp, log1), capture_output=True, text=True)
    check('install serving deps from wheelhouse/serve with --no-index', pip.returncode == 0 and not blocked(log1), pip.stderr.strip()[-200:])
    has_torch = subprocess.run([str(venv / 'bin' / 'python'), '-c', 'import torch'], capture_output=True).returncode == 0
    check('that environment has no PyTorch / timm installed at all', not has_torch)

    # 2. run the service from that venv, under the guard
    log2, data = tmp / 'server.log', tmp / 'data'
    server_out = open(tmp / 'server.stdout', 'w')
    server = subprocess.Popen([str(venv / 'bin' / 'python'), '-m', 'uvicorn', 'tile_service.api:app', '--host', '127.0.0.1', '--port', str(PORT)],
                              cwd=ROOT, env=guarded_env(tmp, log2, TILE_DATA_DIR=str(data), TILE_ORT_THREADS='4'),
                              stdout=server_out, stderr=subprocess.STDOUT)
    try:
        c = httpx.Client(base_url=f'http://127.0.0.1:{PORT}', timeout=30, trust_env=False)
        for _ in range(60):
            try:
                if c.get('/healthz').status_code in (200, 503): break
            except httpx.TransportError:
                time.sleep(0.5)
        h = c.get('/healthz')
        check('service starts offline; artifact checksums + self-test pass', h.status_code == 200 and h.json()['self_test']['passed'],
              f"model {h.json().get('model_version')}")

        first = {}
        for row in csv.DictReader(open(DATASET / 'eval_labels.csv')):
            first.setdefault(row['true_label'], row['filename'])
        results = {}
        for label, name in first.items():
            r = c.post('/v1/tiles', files={'file': (name, (DATASET / 'eval_set' / name).read_bytes(), 'image/png')})
            results[label] = (r.status_code, r.json().get('label'), r.json().get('status'), r.json().get('latency_ms'))
        ok = all(code == 201 for code, *_ in results.values())
        correct = sum(lbl == true for true, (_, lbl, *_) in results.items())
        check('classify one eval tile per class (POST /v1/tiles -> 201)', ok and correct >= 6,
              f'{correct}/7 correct; ' + ', '.join(f'{t}->{l} ({s}, {ms:.0f} ms)' for t, (_, l, s, ms) in results.items()))
        name = first['Forest']
        again = c.post('/v1/tiles', files={'file': (name, (DATASET / 'eval_set' / name).read_bytes(), 'image/png')})
        check('re-post is an idempotent hit (200)', again.status_code == 200 and again.json()['idempotent_hit'])
        tid = again.json()['tile_id']
        check('stored result reads back (GET /v1/tiles, /v1/predictions)',
              c.get(f'/v1/tiles/{tid}').status_code == 200 and len(c.get('/v1/predictions').json()) == 7)
        html = c.get('/docs').text
        ext = sorted(set(re.findall(r'https?://[^\s"\'<>]+', html)))
        assets = [c.get(u).status_code for u in ('/static/swagger-ui/swagger-ui-bundle.js', '/static/swagger-ui/swagger-ui.css', '/openapi.json')]
        check('/docs page references no external URL and its assets are served locally', not ext and assets == [200, 200, 200],
              f'external URLs: {ext or "none"}')

        # 3. every socket owned by the server process must be local
        ss = subprocess.run(['ss', '-tuanp'], capture_output=True, text=True).stdout.splitlines()
        mine = [l for l in ss if f'pid={server.pid},' in l]
        addrs = sorted({tok for l in mine for tok in l.split()[4:6]})  # columns: Netid State Recv-Q Send-Q Local Peer Process
        non_local = [a for a in addrs if not re.match(r'^(127\.\d+\.\d+\.\d+|\[::1\]|\[::ffff:127\.\d+\.\d+\.\d+\]|0\.0\.0\.0|\*|\[::\]):', a)]
        check("all of the server's sockets are on localhost (ss -tuanp)", mine and not non_local, f'{len(mine)} sockets: {addrs}')
    finally:
        server.terminate()
        server.wait(timeout=30)
        server_out.close()

    # 4. nothing in the service tried to reach the network
    ev2 = events(log2)
    check('guard was active in the server process', any(e['event'] == 'guard_loaded' and e['pid'] == server.pid for e in ev2))
    local = [e for e in ev2 if e['pid'] == server.pid and e['event'].startswith('local_')]
    check('server made zero network attempts (none blocked, no local/proxy connects)', not blocked(log2) and not local,
          json.dumps(blocked(log2) + local)[:300])

    # 5. retraining offline: vendored backbone weights, empty caches, same guard
    if not args.skip_train:
        log5 = tmp / 'train.log'
        t0 = time.time()
        tr = subprocess.run([sys.executable, str(ROOT / 'training' / 'train.py'), '--epochs', '1', '--threads', '16', '--no-activate',
                             '--models-dir', str(tmp / 'models')], cwd=ROOT, env=guarded_env(tmp, log5), capture_output=True, text=True)
        manifests = list((tmp / 'models').glob('*/manifest.json'))
        sha_ok = False
        if manifests:
            m = json.loads(manifests[0].read_text())
            vendored = json.loads((ROOT / 'weights' / 'manifest.json').read_text())[m['backbone']['timm_name']]['sha256']
            sha_ok = m['backbone']['pretrained_weights']['sha256'] == vendored
        check('offline retraining (1 epoch) from vendored weights succeeds', tr.returncode == 0 and manifests and sha_ok,
              f'{time.time() - t0:.0f}s; pretrained sha256 matches weights/manifest.json: {sha_ok}' + ('' if tr.returncode == 0 else ' | ' + tr.stderr[-300:]))
        local5 = [e for e in events(log5) if e['event'].startswith('local_')]
        check('training made zero network attempts (guard active, none blocked, no local connects)',
              not blocked(log5) and not local5 and any(e['event'] == 'guard_loaded' for e in events(log5)), json.dumps(blocked(log5) + local5)[:300])
        check('empty Hugging Face / torch caches stayed empty', not any((tmp / d).exists() and any((tmp / d).rglob('*'))
                                                                        for d in ('empty-hf-cache', 'empty-torch-cache')))

    passed = all(ok for _, ok, _ in checks)
    lines = ['# Offline check\n',
             f"Run {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} by `python scripts/offline_check.py`. "
             'Every Python process ran with `scripts/netguard/sitecustomize.py` (blocks and logs any non-localhost DNS lookup, '
             'connection or UDP send; allowed localhost connections are logged too), `HF_HUB_OFFLINE=1`, no proxy variables, and empty Hugging Face / torch caches.\n',
             f"**Result: {'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'}** ({sum(ok for _, ok, _ in checks)}/{len(checks)})\n",
             '| # | Check | Result | Detail |', '|---|---|---|---|']
    lines += [f"| {i} | {n} | {'PASS' if ok else '**FAIL**'} | {d.replace('|', '/')} |" for i, (n, ok, d) in enumerate(checks, 1)]
    (ROOT / 'reports' / 'offline_check.md').write_text('\n'.join(lines) + '\n')
    shutil.rmtree(tmp, ignore_errors=True)
    print('\n' + ('ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED') + ' -> reports/offline_check.md')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()

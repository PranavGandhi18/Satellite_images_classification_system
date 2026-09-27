"""The ONLY step that needs internet. Run once on a connected machine, then carry the repo to the offline box.

Downloads:
  weights/<timm_name>/model.safetensors   pretrained backbones for training/train.py (the original upstream files)
  weights/<timm_name>/LICENSE*            upstream licence files shipped alongside the weights
  tile_service/static/swagger-ui/*        Swagger UI assets, so /docs works without a CDN
and records SHA-256 checksums in weights/manifest.json and tile_service/static/swagger-ui/manifest.json.

The service itself needs none of this at runtime except the Swagger UI files: its model is models/<version>/,
produced by train.py.
usage: python scripts/fetch_offline_assets.py
"""
import datetime as dt
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.backbones import BACKBONES, WEIGHTS_DIR, weights_path  # noqa: E402

SWAGGER_VERSION = '5.33.0'
SWAGGER_DIR = ROOT / 'tile_service' / 'static' / 'swagger-ui'
SWAGGER_FILES = ['swagger-ui-bundle.js', 'swagger-ui.css', 'favicon-32x32.png', 'LICENSE']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_backbones():
    from huggingface_hub import HfApi, hf_hub_download
    manifest = {}
    for key, (timm_name, _, licence) in BACKBONES.items():
        repo = f'timm/{timm_name}'
        dst = weights_path(timm_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(repo, 'model.safetensors'), dst)
        extras = [f for f in HfApi().list_repo_files(repo) if f.upper().startswith('LICENSE') or f in ('config.json', 'README.md')]
        for f in extras:
            shutil.copyfile(hf_hub_download(repo, f), dst.parent / f)
        manifest[timm_name] = dict(key=key, file=str(dst.relative_to(ROOT)), sha256=sha256(dst), bytes=dst.stat().st_size,
                                   source=f'https://huggingface.co/{repo}', licence=licence, extra_files=extras,
                                   downloaded_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'))
        print(f'{timm_name}: {dst.stat().st_size / 1e6:.1f} MB  sha256 {manifest[timm_name]["sha256"][:16]}...  (+ {extras})')
    (WEIGHTS_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def fetch_swagger():
    SWAGGER_DIR.mkdir(parents=True, exist_ok=True)
    manifest = dict(package='swagger-ui-dist', version=SWAGGER_VERSION, licence='Apache-2.0', files={})
    for name in SWAGGER_FILES:
        url = f'https://cdn.jsdelivr.net/npm/swagger-ui-dist@{SWAGGER_VERSION}/{name}'
        with urllib.request.urlopen(url, timeout=60) as r:
            (SWAGGER_DIR / name).write_bytes(r.read())
        manifest['files'][name] = dict(source=url, sha256=sha256(SWAGGER_DIR / name))
        print(f'swagger-ui {SWAGGER_VERSION}: {name} ({(SWAGGER_DIR / name).stat().st_size / 1e3:.0f} KB)')
    (SWAGGER_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    fetch_backbones()
    fetch_swagger()

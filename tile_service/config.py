"""Runtime settings, read from environment variables (all optional)."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_dir: Path = ROOT / 'models' / 'active'  # a models/<version>/ directory (or the "active" symlink)
    data_dir: Path = ROOT / 'data'                 # holds tiles.db and blobs/
    ort_threads: int = 4                           # ONNX Runtime intra-op threads per inference
    max_upload_bytes: int = 1_000_000              # a 64x64 PNG is ~2-10 KB; anything this big is not a tile
    access_key: str | None = None                  # if set, every request must present it (see api.py); off by default

    @property
    def db_path(self) -> Path:
        return self.data_dir / 'tiles.db'

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / 'blobs'

    @classmethod
    def from_env(cls) -> 'Settings':
        d = cls()
        return cls(model_dir=Path(os.environ.get('TILE_MODEL_DIR', d.model_dir)),
                   data_dir=Path(os.environ.get('TILE_DATA_DIR', d.data_dir)),
                   ort_threads=int(os.environ.get('TILE_ORT_THREADS', d.ort_threads)),
                   max_upload_bytes=int(os.environ.get('TILE_MAX_UPLOAD_BYTES', d.max_upload_bytes)),
                   access_key=_read_key())


def _read_key() -> str | None:
    """TILE_ACCESS_KEY, or the contents of the file named by TILE_ACCESS_KEY_FILE (keeps the key out of process listings)."""
    key = os.environ.get('TILE_ACCESS_KEY')
    if not key and os.environ.get('TILE_ACCESS_KEY_FILE'):
        key = Path(os.environ['TILE_ACCESS_KEY_FILE']).read_text().strip()
    if key and len(key) < 16:
        raise ValueError('TILE_ACCESS_KEY must be at least 16 characters')
    return key or None

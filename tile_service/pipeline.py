"""The tile flow from DESIGN_NOTE.md §2 as one function: validate -> hash -> (idempotency) -> quality flags ->
store raw tile -> classify -> score -> persist. The HTTP endpoint calls it; a batch/folder-drop job would call the
same thing, so there is exactly one code path to test.
"""
import json
import logging
import time
from dataclasses import dataclass

from .scoring import score
from .storage import BlobStore, Repository, utcnow
from .validation import HEIGHT, WIDTH, decode_and_validate, quality_flags

log = logging.getLogger('tile_service')


@dataclass(frozen=True)
class TileMetadata:
    source_name: str | None = None
    captured_at: str | None = None
    lat: float | None = None
    lon: float | None = None
    sensor: str | None = None


@dataclass(frozen=True)
class ProcessResult:
    record: dict   # the stored result (current prediction joined with its tile)
    created: bool  # False = idempotent hit: this exact tile was already classified by this model version


class TilePipeline:
    def __init__(self, model, repo: Repository, blobs: BlobStore, max_upload_bytes: int):
        self.model, self.repo, self.blobs, self.max_upload_bytes = model, repo, blobs, max_upload_bytes

    def process(self, data: bytes, meta: TileMetadata) -> ProcessResult:
        tile = decode_and_validate(data, self.max_upload_bytes)  # raises InvalidTile -> HTTP 400

        existing = self.repo.current_result(tile.sha256, self.model.version)
        if existing and existing['status'] != 'failed':  # a failed attempt is retried, anything else is returned as-is
            return ProcessResult(existing, created=False)

        flags = quality_flags(tile.pixels)
        blob_path = self.blobs.put(tile.sha256, data)  # written before the DB row, so a row never points at a missing file
        now = utcnow()
        tile_row = dict(tile_id=tile.sha256, source_name=meta.source_name, received_at=now, captured_at=meta.captured_at,
                        lat=meta.lat, lon=meta.lon, sensor=meta.sensor, width=WIDTH, height=HEIGHT, bands=3,
                        byte_size=tile.byte_size, quality_flags=json.dumps(flags), blob_path=blob_path)
        pred = dict(tile_id=tile.sha256, model_version=self.model.version, policy_version=self.model.policy['version'],
                    created_at=now, label=None, confidence=None, margin=None, probabilities=None, ood_score=None, error=None)

        t0 = time.perf_counter()
        try:
            out = self.model.infer(tile.pixels)
            s = score(out.logits, out.embedding, classes=self.model.classes, temperature=self.model.temperature,
                      references=self.model.references, policy=self.model.policy, flags=flags)
            pred.update(status=s.status, label=s.label, confidence=s.confidence, margin=s.margin,
                        probabilities=json.dumps(s.probabilities), ood_score=s.ood_score)
        except Exception as e:  # the tile is still stored, marked failed, and retried on the next upload
            log.exception('inference failed for tile %s', tile.sha256)
            pred.update(status='failed', error=f'{type(e).__name__}: {e}'[:500])
        pred['latency_ms'] = round((time.perf_counter() - t0) * 1000, 3)

        self.repo.save(tile_row, pred)
        log.info('tile=%s status=%s label=%s conf=%s ood=%s latency_ms=%.1f flags=%s', tile.sha256[:12], pred['status'],
                 pred['label'], pred['confidence'] and round(pred['confidence'], 4),
                 pred['ood_score'] and round(pred['ood_score'], 4), pred['latency_ms'], flags)
        return ProcessResult(self.repo.current_result(tile.sha256, self.model.version), created=True)

"""Response shapes (also what shows up in the OpenAPI docs at /docs)."""
from pydantic import BaseModel, Field


class TileResult(BaseModel):
    tile_id: str
    model_version: str
    policy_version: str
    status: str                       # confident | needs_review | unfamiliar | failed
    label: str | None = None
    confidence: float | None = None   # calibrated probability of the label
    margin: float | None = None       # top-1 minus top-2 probability
    probabilities: dict[str, float] | None = None
    ood_score: float | None = None    # 1 - cosine similarity to the nearest training tile
    quality_flags: list[str]
    latency_ms: float | None = None
    error: str | None = None
    created_at: str
    source_name: str | None = None
    received_at: str
    captured_at: str | None = None
    lat: float | None = None
    lon: float | None = None
    sensor: str | None = None
    byte_size: int
    review_label: str | None = None   # latest analyst label, if the tile has been reviewed
    reviewer: str | None = None
    reviewed_at: str | None = None


class QueueItem(TileResult):
    reason: str                       # why this tile is in the review queue


class ReviewIn(BaseModel):
    label: str = Field(description="one of the model's classes, or Other (none of them) / Unusable (cloud, no-data, ...)")
    note: str | None = Field(None, max_length=500)
    reviewer: str | None = Field(None, max_length=64)


class IngestResult(TileResult):
    idempotent_hit: bool              # true = this exact tile was already classified by this model; nothing was re-run


class Health(BaseModel):
    status: str                       # ok | degraded
    model_version: str | None = None
    model_error: str | None = None
    self_test: dict | None = None
    database_ok: bool
    disk_free_mb: float

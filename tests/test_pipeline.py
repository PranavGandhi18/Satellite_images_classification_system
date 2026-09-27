import pytest

from tile_service.pipeline import TileMetadata, TilePipeline
from tile_service.storage import BlobStore, Repository
from tile_service.validation import InvalidTile

from .conftest import png_bytes, random_tile


@pytest.fixture
def pipeline(settings, fake_model):
    repo = Repository(settings.db_path)
    repo.register_model(fake_model.manifest)
    return TilePipeline(fake_model, repo, BlobStore(settings.blob_dir), settings.max_upload_bytes)


def test_new_tile_is_stored_classified_and_its_raw_bytes_kept(pipeline):
    data = png_bytes(random_tile(1))
    r = pipeline.process(data, TileMetadata(source_name='a.png', lat=12.5, lon=77.6))
    assert r.created and r.record['status'] == 'confident' and r.record['label']
    assert sum(r.record['probabilities'].values()) == pytest.approx(1, abs=1e-5)
    stored = pipeline.repo.tile_history(r.record['tile_id'])
    assert stored['tile']['lat'] == 12.5 and len(stored['predictions']) == 1
    assert pipeline.blobs.path(stored['tile']['blob_path']).read_bytes() == data


def test_same_tile_twice_is_an_idempotent_hit(pipeline, fake_model):
    data = png_bytes(random_tile(2))
    first = pipeline.process(data, TileMetadata())
    second = pipeline.process(data, TileMetadata())
    assert not second.created and second.record['prediction_id'] == first.record['prediction_id']
    assert fake_model.calls == 1  # the model did not run again


def test_inference_failure_is_recorded_then_retried(pipeline, fake_model):
    data = png_bytes(random_tile(3))
    fake_model.fail = True
    failed = pipeline.process(data, TileMetadata())
    assert failed.record['status'] == 'failed' and 'simulated' in failed.record['error']
    fake_model.fail = False
    retried = pipeline.process(data, TileMetadata())
    assert retried.created and retried.record['status'] == 'confident'
    history = pipeline.repo.tile_history(retried.record['tile_id'])['predictions']
    assert [p['status'] for p in history] == ['failed', 'confident']  # append-only: the failure is kept


def test_invalid_tile_stores_nothing(pipeline):
    with pytest.raises(InvalidTile):
        pipeline.process(b'not a png', TileMetadata())
    assert pipeline.repo.list_results(model_version='fake-v1') == []
    assert not pipeline.blobs.root.exists() or not any(pipeline.blobs.root.rglob('*.png'))


def test_listing_filters(pipeline):
    for seed in range(6):
        pipeline.process(png_bytes(random_tile(10 + seed)), TileMetadata())
    rows = pipeline.repo.list_results(model_version='fake-v1')
    assert len(rows) == 6
    label = rows[0]['label']
    assert all(r['label'] == label for r in pipeline.repo.list_results(model_version='fake-v1', label=label))
    assert pipeline.repo.list_results(model_version='fake-v1', status='unfamiliar') == []

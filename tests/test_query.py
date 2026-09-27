"""Querying the results (DESIGN_NOTE.md §4.3): lookup, filter, aggregate, review queue, labels, CSV export."""
import csv
import io
import json
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tile_service.api import create_app
from tile_service.model import RawOutput

from .conftest import CLASSES, FakeModel, png_bytes


class QueryFake(FakeModel):
    """FakeModel whose tiles with pixel[0,0,0] == 1 come out far from all training embeddings (-> unfamiliar)."""
    def infer(self, pixels):
        out = super().infer(pixels)
        if pixels[0, 0, 0] == 1:
            return RawOutput(logits=out.logits, embedding=np.ones(8, dtype=np.float32))
        return out


def tile(k: int, seed: int = 0, odd: bool = False) -> np.ndarray:
    """A textured, dark tile that FakeModel classifies as CLASSES[k] (mean exactly 42 + k, and 42 % 7 == 0) and that
    trips no quality flag. Different seeds give different bytes (so no idempotent hits) with the same mean."""
    a = np.full((64, 64, 3), 42 + k, np.int16)
    a[::2] += 3; a[1::2] -= 3
    a[2, seed % 64, 1] += 1; a[3, seed % 64, 1] -= 1
    if odd:                                   # mark as "unfamiliar" for QueryFake, keeping the mean
        delta = int(a[0, 0, 0]) - 1
        a[0, 0, 0] = 1; a[0, 2, 0] += delta
    return a.astype(np.uint8)


def up(c, arr, **form):
    return c.post('/v1/tiles', files={'file': ('t.png', png_bytes(arr), 'image/png')}, data=form)


@pytest.fixture
def c(settings):
    with TestClient(create_app(settings, model=QueryFake())) as client:
        yield client


@pytest.fixture
def populated(c):
    for k in range(7):
        for s in range(k + 1):                       # class k gets k+1 tiles
            assert up(c, tile(k, s)).status_code == 201
    assert up(c, np.full((64, 64, 3), 120, np.uint8)).json()['status'] == 'needs_review'   # blank -> needs_review
    assert up(c, tile(3, 99, odd=True)).json()['status'] == 'unfamiliar'
    return c


def test_filter_label_status_flag_search_sort_and_paging(populated):
    c = populated
    r = c.get('/v1/predictions', params={'label': 'SeaLake', 'limit': 3})
    assert r.headers['X-Total-Count'] == '7' and len(r.json()) == 3 and {x['label'] for x in r.json()} == {'SeaLake'}
    assert c.get('/v1/predictions', params={'label': 'SeaLake', 'limit': 3, 'offset': 6}).json().__len__() == 1
    assert c.get('/v1/predictions', params={'status': 'unfamiliar'}).headers['X-Total-Count'] == '1'
    assert c.get('/v1/predictions', params={'flag': 'blank'}).headers['X-Total-Count'] == '1'
    assert c.get('/v1/predictions', params={'flag': 'none'}).headers['X-Total-Count'] == '29'
    one = c.get('/v1/predictions', params={'limit': 1}).json()[0]
    assert c.get('/v1/predictions', params={'q': one['tile_id'][:10]}).json()[0]['tile_id'] == one['tile_id']
    ood = [x['ood_score'] for x in c.get('/v1/predictions', params={'sort': 'ood_desc', 'limit': 500}).json()]
    assert ood == sorted(ood, reverse=True)
    for bad in ({'label': 'Moon'}, {'bbox': '1,2,3'}, {'received_from': 'yesterday'}, {'sort': 'random'}):
        assert c.get('/v1/predictions', params=bad).status_code in (400, 422), bad


def test_time_and_area_filters(c):
    up(c, tile(0, 1), lat='48.1', lon='11.5', captured_at='2026-06-01T12:00:00+02:00')
    up(c, tile(0, 2))
    today, yesterday = date.today().isoformat(), (date.today() - timedelta(days=1)).isoformat()
    assert c.get('/v1/predictions', params={'received_from': today}).headers['X-Total-Count'] == '2'
    assert c.get('/v1/predictions', params={'received_to': yesterday}).headers['X-Total-Count'] == '0'
    inside = c.get('/v1/predictions', params={'bbox': '11,48,12,49'}).json()
    assert len(inside) == 1 and inside[0]['lat'] == 48.1                       # the tile without coordinates never matches
    assert inside[0]['captured_at'] == '2026-06-01T10:00:00.000+00:00'          # stored normalised to UTC
    assert c.get('/v1/predictions', params={'captured_from': '2026-06-01', 'captured_to': '2026-06-01'}).headers['X-Total-Count'] == '1'


def test_aggregate_raw_and_corrected(populated, settings):
    s = populated.get('/v1/stats').json()
    raw = {r['label']: r['raw_count'] for r in s['by_class']}
    assert s['n_classified'] == 30 and s['by_status']['unfamiliar'] == 1 and s['by_status']['needs_review'] == 1
    # class k got k+1 tiles; the blank grey tile (120 % 7 = 1) adds a Forest, the unfamiliar tile an Industrial
    assert raw == {'AnnualCrop': 1, 'Forest': 3, 'Highway': 3, 'Industrial': 5, 'Residential': 5, 'River': 6, 'SeaLake': 7}
    assert s['correction']['available'] is False and 'calibration' in s['correction']['reason']
    # with a calibration file (as written by training/evaluate.py) the corrected shares appear
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    conf = (np.eye(7) * 28 + 2 * (np.ones((7, 7)) - np.eye(7)) / 6).round().astype(int)
    (settings.model_dir / 'aggregate_calibration.json').write_text(json.dumps(
        dict(model_version='fake-v1', classes=CLASSES, confusion_counts=conf.tolist(), n=int(conf.sum()), source='test')))
    s = populated.get('/v1/stats', params={'label': ''}).json()      # an empty form field means 'no filter'
    assert s['correction']['available'] is True
    shares = [r['corrected_share'] for r in s['by_class']]
    assert abs(sum(shares) - 1) < 1e-6
    assert all(r['corrected_low'] <= r['corrected_share'] + 1e-9 <= r['corrected_high'] + 2e-9 for r in s['by_class'])


def test_review_queue_labels_and_lookup(populated):
    c = populated
    q = c.get('/v1/review-queue')
    items = q.json()
    assert q.headers['X-Total-Count'] == '2' and [i['status'] for i in items] == ['unfamiliar', 'needs_review']
    assert 'OOD score' in items[0]['reason'] and 'blank' in items[1]['reason']
    assert c.post(f"/v1/tiles/{items[0]['tile_id']}/label", json={'label': 'Moon'}).status_code == 400
    assert c.post('/v1/tiles/' + '0' * 64 + '/label', json={'label': 'Forest'}).status_code == 404
    r = c.post(f"/v1/tiles/{items[0]['tile_id']}/label", json={'label': 'Other', 'note': 'looks like a quarry', 'reviewer': 'qa'})
    assert r.status_code == 201 and r.json()['label'] == 'Other'
    assert c.get('/v1/review-queue').headers['X-Total-Count'] == '1'          # labelled tile left the queue
    c.post(f"/v1/tiles/{items[1]['tile_id']}/label", json={'label': 'Unusable'})
    assert c.get('/v1/review-queue').json() == []
    hist = c.get(f"/v1/tiles/{items[0]['tile_id']}").json()
    assert hist['reviews'][0]['note'] == 'looks like a quarry'
    listed = c.get('/v1/predictions', params={'reviewed': True}).json()
    assert {x['review_label'] for x in listed} == {'Other', 'Unusable'}
    rv = c.get('/v1/stats').json()['reviews']
    assert rv['n'] == 2 and rv['agree'] == 0 and rv['by_label'] == {'Other': 1, 'Unusable': 1}


def test_csv_export_image_and_meta(populated):
    c = populated
    r = c.get('/v1/predictions.csv', params={'label': 'SeaLake'})
    assert r.headers['content-type'].startswith('text/csv') and 'attachment' in r.headers['content-disposition']
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == 7 and {x['label'] for x in rows} == {'SeaLake'} and 'p_SeaLake' in rows[0]
    data = png_bytes(tile(2, 5))
    tid = c.post('/v1/tiles', files={'file': ('x.png', data, 'image/png')}).json()['tile_id']
    img = c.get(f'/v1/tiles/{tid}/image')
    assert img.status_code == 200 and img.content == data and img.headers['content-type'] == 'image/png'
    assert c.get('/v1/tiles/' + 'f' * 64 + '/image').status_code == 404
    m = c.get('/v1/meta').json()
    assert m['classes'] == CLASSES and m['review_labels'][-2:] == ['Other', 'Unusable'] and 'min_confidence' in m['policy']
    assert c.get('/explore').status_code == 200

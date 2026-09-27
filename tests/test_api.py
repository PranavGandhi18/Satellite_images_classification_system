import pytest
from fastapi.testclient import TestClient

from tile_service.api import create_app

from .conftest import png_bytes, random_tile


@pytest.fixture
def client(settings, fake_model):
    with TestClient(create_app(settings, model=fake_model)) as c:
        yield c


def post(client, data, **form):
    return client.post('/v1/tiles', files={'file': ('tile.png', data, 'image/png')}, data=form)


def test_post_tile_classifies_and_stores(client):
    r = post(client, png_bytes(random_tile(1)), lat='12.97', lon='77.59', sensor='S2', captured_at='2026-09-01T10:30:00Z')
    assert r.status_code == 201, r.text
    body = r.json()
    assert body['status'] == 'confident' and body['label'] and body['idempotent_hit'] is False
    assert body['model_version'] == 'fake-v1' and body['sensor'] == 'S2' and body['lat'] == 12.97
    assert set(body['probabilities']) == {'AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake'}

    got = client.get(f"/v1/tiles/{body['tile_id']}")
    assert got.status_code == 200 and got.json()['predictions'][0]['label'] == body['label']


def test_reposting_the_same_tile_returns_200_and_the_stored_result(client):
    data = png_bytes(random_tile(2))
    first, second = post(client, data), post(client, data)
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()['idempotent_hit'] is True and second.json()['created_at'] == first.json()['created_at']


@pytest.mark.parametrize('data, form, code', [
    (b'garbage', {}, 400),
    (png_bytes(random_tile()[:32, :32]), {}, 400),
    (png_bytes(random_tile()), {'lat': '123'}, 422),
    (png_bytes(random_tile()), {'captured_at': 'yesterday'}, 400),
])
def test_bad_requests(client, data, form, code):
    assert post(client, data, **form).status_code == code


def test_inference_failure_returns_500_with_the_stored_record(client, fake_model):
    fake_model.fail = True
    r = post(client, png_bytes(random_tile(4)))
    assert r.status_code == 500 and r.json()['status'] == 'failed'


def test_listing_unknown_tile_and_health(client):
    post(client, png_bytes(random_tile(5)))
    assert len(client.get('/v1/predictions').json()) == 1
    assert client.get('/v1/predictions', params={'status': 'bogus'}).status_code == 422
    assert client.get('/v1/tiles/' + '0' * 64).status_code == 404
    h = client.get('/healthz')
    assert h.status_code == 200 and h.json()['status'] == 'ok'


def test_missing_model_degrades_instead_of_crashing(settings):
    with TestClient(create_app(settings)) as c:  # settings.model_dir does not exist
        h = c.get('/healthz')
        assert h.status_code == 503 and 'manifest.json' in h.json()['model_error']
        assert post(c, png_bytes(random_tile())).status_code == 503


def test_docs_page_is_self_contained(client):
    import re
    html = client.get('/docs').text
    assert re.findall(r'https?://[^\s"\'<>]+', html) == []  # no CDN: must work on an air-gapped machine
    for asset in ('swagger-ui-bundle.js', 'swagger-ui.css', 'favicon-32x32.png'):
        assert client.get(f'/static/swagger-ui/{asset}').status_code == 200


def test_test_page_and_samples_are_self_contained(client):
    import json, re
    html = client.get('/').text
    assert 'Tile Classifier' in html and re.findall(r'https?://[^\s"\'<>]+', html) == []
    explore = client.get('/explore').text
    assert 'Query stored results' in explore and re.findall(r'https?://[^\s"\'<>]+', explore) == []
    samples = client.get('/static/ui/samples.json').json()
    assert len(samples) >= 14
    for s in samples:
        r = client.get(f"/static/ui/samples/{s['file']}")
        assert r.status_code == 200 and r.content[:4] == b'\x89PNG'
    assert post(client, client.get(f"/static/ui/samples/{samples[0]['file']}").content).status_code == 201


KEY = 'test-key-0123456789abcdef'


@pytest.fixture
def locked_client(settings, fake_model):
    from dataclasses import replace
    with TestClient(create_app(replace(settings, access_key=KEY), model=fake_model)) as c:
        yield c


def test_access_key_blocks_everything_without_it(locked_client):
    for method, path in (('get', '/'), ('get', '/healthz'), ('get', '/docs'), ('get', '/static/ui/samples.json'), ('get', '/v1/predictions')):
        assert getattr(locked_client, method)(path).status_code == 401, path
    assert post(locked_client, png_bytes(random_tile())).status_code == 401
    assert locked_client.get('/healthz', headers={'X-API-Key': 'wrong-key-0123456789'}).status_code == 401
    page = locked_client.get('/', headers={'Accept': 'text/html'})
    assert page.status_code == 401 and 'Access key required' in page.text


def test_access_key_via_header_bearer_and_shared_link(locked_client):
    assert locked_client.get('/healthz', headers={'X-API-Key': KEY}).status_code == 200
    assert locked_client.get('/healthz', headers={'Authorization': f'Bearer {KEY}'}).status_code == 200
    r = locked_client.get(f'/?key={KEY}&x=1', follow_redirects=False)            # shared link
    assert r.status_code == 303 and r.headers['location'] == '/?x=1'               # key stripped from the URL...
    assert 'tile_access=' in r.headers['set-cookie'] and 'httponly' in r.headers['set-cookie'].lower()
    assert locked_client.get('/').status_code == 200                               # ...and the cookie carries it
    assert post(locked_client, png_bytes(random_tile(9))).status_code == 201


def test_health_dashboard_and_ops_metrics(client):
    import re
    post(client, png_bytes(random_tile(21)))
    html = client.get('/health').text
    assert 'Service health' in html and re.findall(r'https?://[^\s"\'<>]+', html) == []   # self-contained, works offline
    o = client.get('/v1/ops').json()
    assert o['status'] == 'good' and {c['name'] for c in o['checks']} >= {'Model artifact', 'Self-test', 'Database', 'Disk space'}
    assert all(c['status'] in ('good', 'warning', 'serious', 'critical') for c in o['checks'])
    assert o['daily'][-1]['n'] >= 1 and o['latency_ms']['n'] >= 1 and o['class_mix']['n'] >= 1 and o['totals']['tiles'] >= 1
    assert o['model']['classes'] and 0 < o['model']['expected_unfamiliar_rate'] < 0.05


def test_pages_are_never_served_stale(client):
    for path in ('/', '/explore', '/health'):
        assert client.get(path).headers['cache-control'] == 'no-cache', path

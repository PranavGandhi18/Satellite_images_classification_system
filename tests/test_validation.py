import numpy as np
import pytest

from tile_service.validation import InvalidTile, decode_and_validate, quality_flags

from .conftest import DATASET, png_bytes, random_tile


def test_valid_tile_decodes_and_is_hashed_by_content():
    data = png_bytes(random_tile())
    t = decode_and_validate(data, max_bytes=1_000_000)
    assert t.pixels.shape == (64, 64, 3) and t.pixels.dtype == np.uint8
    assert len(t.sha256) == 64 and t.byte_size == len(data)
    assert decode_and_validate(data, 1_000_000).sha256 == t.sha256


@pytest.mark.parametrize('data, message', [
    (b'', 'empty'),
    (b'definitely not an image', 'not a decodable image'),
    (png_bytes(np.zeros((32, 32, 3), np.uint8)), 'expected 64x64'),
    (png_bytes(np.zeros((64, 64), np.uint8)), 'expected 3-band'),
    (png_bytes(np.zeros((64, 64, 4), np.uint8)), 'expected 3-band'),
    (png_bytes(random_tile(), fmt='JPEG'), 'expected a PNG'),
])
def test_invalid_uploads_are_rejected_with_a_reason(data, message):
    with pytest.raises(InvalidTile, match=message):
        decode_and_validate(data, max_bytes=1_000_000)


def test_oversize_upload_is_rejected_before_decoding():
    with pytest.raises(InvalidTile, match='limit'):
        decode_and_validate(b'x' * 2001, max_bytes=2000)


def test_truncated_png_is_rejected():
    data = png_bytes(random_tile())
    with pytest.raises(InvalidTile):
        decode_and_validate(data[: len(data) // 2], max_bytes=1_000_000)


def test_quality_flags():
    assert quality_flags(np.full((64, 64, 3), 120, np.uint8)) == ['blank']
    grey = np.clip(110 + np.random.default_rng(0).normal(0, 3, (64, 64, 3)), 0, 255).astype(np.uint8)
    assert 'grey_low_texture' in quality_flags(grey)
    assert 'saturated' in quality_flags(np.full((64, 64, 3), 255, np.uint8))
    assert 'nodata_pixels' in quality_flags(np.zeros((64, 64, 3), np.uint8))
    assert quality_flags(random_tile()) == []


@pytest.mark.skipif(not DATASET.exists(), reason='dataset not present')
def test_real_tiles_validate_and_the_known_grey_tile_is_flagged():
    for p in sorted((DATASET / 'candidate_tiles').glob('*/*_1.png')):
        decode_and_validate(p.read_bytes(), 1_000_000)
    grey = decode_and_validate((DATASET / 'candidate_tiles/SeaLake/SeaLake_18.png').read_bytes(), 1_000_000)
    assert quality_flags(grey.pixels) != []

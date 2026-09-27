"""Step 1 and 3 of the tile flow: strict input validation, then cheap quality flags.

Validation *rejects* anything that is not what the model was trained on (a 64x64, 3-band, 8-bit RGB PNG).
Quality checks never reject; they only flag, so that the stored result says why a tile may be unreliable.
"""
import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

WIDTH = HEIGHT = 64
Image.MAX_IMAGE_PIXELS = 4096 * 4096  # decompression-bomb guard; the header size check below rejects anything bigger than 64x64 first


class InvalidTile(ValueError):
    """The upload is not a tile this service can classify. The message is safe to return to the client."""


@dataclass(frozen=True)
class DecodedTile:
    pixels: np.ndarray  # (64, 64, 3) uint8, RGB
    sha256: str         # of the uploaded bytes: the tile's identity (idempotency key and blob address)
    byte_size: int


def decode_and_validate(data: bytes, max_bytes: int) -> DecodedTile:
    if not data:
        raise InvalidTile('empty upload')
    if len(data) > max_bytes:
        raise InvalidTile(f'upload is {len(data)} bytes; the limit is {max_bytes}')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != 'PNG':
                raise InvalidTile(f'expected a PNG, got {im.format or "an unknown format"}')
            if im.size != (WIDTH, HEIGHT):
                raise InvalidTile(f'expected {WIDTH}x{HEIGHT} pixels, got {im.size[0]}x{im.size[1]}')
            if im.mode != 'RGB':
                raise InvalidTile(f'expected 3-band 8-bit RGB, got PIL mode {im.mode!r}')
            im.load()
            pixels = np.array(im, dtype=np.uint8)
    except InvalidTile:
        raise
    except UnidentifiedImageError:
        raise InvalidTile('not a decodable image') from None
    except (OSError, SyntaxError, ValueError) as e:
        raise InvalidTile(f'corrupt image: {e}') from None
    return DecodedTile(pixels=pixels, sha256=hashlib.sha256(data).hexdigest(), byte_size=len(data))


def quality_flags(pixels: np.ndarray) -> list[str]:
    """Heuristics from the data analysis (DESIGN_NOTE.md finding 7): flat grey tiles look like haze/cloud/no-data,
    and the model tends to call them SeaLake with high confidence."""
    f = pixels.astype(np.float32)
    texture = float(f.std(axis=(0, 1)).mean())   # within-tile variation, averaged over bands
    chroma = float((f.max(2) - f.min(2)).mean())  # how far from grey the pixels are
    value_range = int((pixels.max(axis=(0, 1)).astype(int) - pixels.min(axis=(0, 1))).max())
    flags = []
    # "blank" = a constant fill-value tile. It must not be a low-texture threshold: calm deep water is nearly uniform
    # (37 of the 150 SeaLake training tiles have texture < 1), but no real training tile has a value range below 6.
    if value_range <= 2:
        flags.append('blank')
    elif texture < 6.0 and chroma < 25.0 and f.mean() > 80.0:
        flags.append('grey_low_texture')
    if (pixels == 255).mean() > 0.05:
        flags.append('saturated')
    if (pixels == 0).all(axis=2).mean() > 0.05:
        flags.append('nodata_pixels')
    return flags

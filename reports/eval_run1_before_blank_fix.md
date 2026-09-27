# Evaluation report

- Model: `dinov3_vits16-112px-20260926-182432`
- Mode: HTTP against http://127.0.0.1:8000
- Eval tiles: 210 (sent in shuffled order; eval_set was not used for any training or threshold choice)

## Accuracy

| Measure | Value |
|---|---|
| Accuracy, all tiles (label regardless of status) | **98.6%** (95% CI 95.9%–99.5%) |
| Tiles marked `confident` (coverage) | 91.9% |
| Accuracy on `confident` tiles | **99.5%** |
| Model latency per tile, p50 / p95 | 21.5 / 25.7 ms |
| Full request round trip, p50 / p95 | 29.5 / 34.4 ms |

| Status | Tiles | Share | Accuracy |
|---|---|---|---|
| confident | 193 | 91.9% | 99.5% |
| needs_review | 11 | 5.2% | 100.0% |
| unfamiliar | 6 | 2.9% | 66.7% |

**Per-class recall:** AnnualCrop 100.0%, Forest 100.0%, Highway 100.0%, Industrial 96.7%, Residential 100.0%, River 96.7%, SeaLake 96.7%

**Confusion matrix** (rows = true, columns = predicted):

| | AnnualCrop | Forest | Highway | Industrial | Residential | River | SeaLake |
|---|---|---|---|---|---|---|---|
| **AnnualCrop** | 30 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Forest** | 0 | 30 | 0 | 0 | 0 | 0 | 0 |
| **Highway** | 0 | 0 | 30 | 0 | 0 | 0 | 0 |
| **Industrial** | 0 | 0 | 1 | 29 | 0 | 0 | 0 |
| **Residential** | 0 | 0 | 0 | 0 | 30 | 0 | 0 |
| **River** | 0 | 0 | 1 | 0 | 0 | 29 | 0 |
| **SeaLake** | 1 | 0 | 0 | 0 | 0 | 0 | 29 |

**Misclassified tiles:**

| File | True | Predicted | Status | Confidence | OOD score |
|---|---|---|---|---|---|
| tile_118.png | Industrial | Highway | unfamiliar | 0.519 | 0.220 |
| tile_034.png | River | Highway | confident | 0.981 | 0.109 |
| tile_174.png | SeaLake | AnnualCrop | unfamiliar | 0.475 | 0.542 |

## Behaviour checks

- **Idempotency:** re-posting an already-classified tile returned HTTP 200, `idempotent_hit=True`, same stored result: True.
- **Invalid uploads** (all must be rejected with 400 and a reason):

| Upload | HTTP | Reason |
|---|---|---|
| garbage bytes | 400 | not a decodable image |
| 32x32 PNG | 400 | expected 64x64 pixels, got 32x32 |
| JPEG instead of PNG | 400 | expected a PNG, got JPEG |
| grayscale PNG | 400 | expected 3-band 8-bit RGB, got PIL mode 'L' |
| truncated PNG | 400 | corrupt image: image file is truncated |

- **Synthetic tiles unlike anything in training:**

| Tile | Status | Label | Confidence | OOD score | Quality flags |
|---|---|---|---|---|---|
| uniform grey (no-data-like) | needs_review | SeaLake | 0.992 | 0.071 | blank |
| bright hazy white (cloud-like) | needs_review | SeaLake | 0.995 | 0.051 | grey_low_texture |
| random noise | unfamiliar | Residential | 0.768 | 0.337 | – |
| black/white checkerboard | unfamiliar | AnnualCrop | 0.850 | 0.378 | saturated, nodata_pixels |
| saturated magenta | unfamiliar | SeaLake | 0.923 | 0.324 | blank, saturated |

- **Real eval tiles that tripped a quality flag:**

| File | True | Predicted | Status | Confidence | OOD score | Flags |
|---|---|---|---|---|---|---|
| tile_105.png | Industrial | Industrial | confident | 0.996 | 0.019 | saturated |
| tile_153.png | SeaLake | SeaLake | needs_review | 0.996 | 0.000 | blank |
| tile_180.png | SeaLake | SeaLake | needs_review | 0.996 | 0.000 | blank |
| tile_098.png | Industrial | Industrial | confident | 0.996 | 0.014 | saturated |
| tile_151.png | SeaLake | SeaLake | needs_review | 0.840 | 0.060 | grey_low_texture |
| tile_096.png | Industrial | Industrial | confident | 0.996 | 0.032 | saturated |
| tile_173.png | SeaLake | SeaLake | needs_review | 0.996 | 0.000 | blank |
| tile_177.png | SeaLake | SeaLake | needs_review | 0.996 | 0.000 | blank |
| tile_154.png | SeaLake | SeaLake | needs_review | 0.996 | 0.001 | blank |
| tile_179.png | SeaLake | SeaLake | needs_review | 0.996 | 0.002 | blank |
| tile_169.png | SeaLake | SeaLake | needs_review | 0.996 | 0.001 | blank |
| tile_172.png | SeaLake | SeaLake | needs_review | 0.996 | 0.001 | blank |
| tile_157.png | SeaLake | SeaLake | needs_review | 0.996 | 0.000 | blank |
| tile_155.png | SeaLake | SeaLake | needs_review | 0.996 | 0.009 | grey_low_texture |
| tile_123.png | AnnualCrop | AnnualCrop | confident | 0.996 | 0.033 | saturated |
| tile_104.png | Industrial | Industrial | confident | 0.996 | 0.010 | saturated |

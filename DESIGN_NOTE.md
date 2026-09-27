# Design Note: Offline Land-Use Classification Service

*GalaxEye take-home, Part 1 (approach). Code comes in Part 2.*

## TL;DR

- **Shape of the system.** A single-node Python service: **ingest → validate → classify → store → query**. Nothing in it needs the network at runtime. The stack is FastAPI, ONNX Runtime on CPU, SQLite, and a content-addressed folder for the raw tiles.
- **Model.** A **ViT-S/16 transformer fine-tuned at 112 px** on the 1,050 labelled `candidate_tiles` (DINOv3 weights if the licence clears; otherwise Apache-2.0 ImageNet-21k weights). It fine-tunes in ~25 minutes on CPU, scores 98.9% on held-out tiles, and runs at ~30 ms per tile as ONNX. Its embeddings also answer "have I seen anything like this before?" The measured comparison of ~58 alternatives is in `model_selection_thoughtprocess.md`.
- **Uncertainty.** Two rules: **never throw a prediction away, and never bake a threshold into the stored data.** I store the full probability vector, a confidence score, an "unfamiliar input" score and input-quality flags. Whether a result is "confident" or "needs review" comes from a versioned policy applied on top, so the policy can change without re-running the model.
- **Traceability.** Every stored result is tied to the exact model version and the input's hash, so any answer can be reproduced, re-scored or audited later.

---

## 1. What the data told me, and what it changes

I analysed the data before designing anything. Appendix A has the method and the numbers.

| # | Finding | Consequence for the design |
|---|---|---|
| 1 | All 1,260 tiles are **64×64 RGB 8-bit PNG**, about 6.5 KB each. At Sentinel-2's 10 m/px, one tile covers about 640 m × 640 m. | Inference is cheap on CPU. Keeping every raw tile is cheap too: 1 M tiles ≈ 6.5 GB. Input validation can be strict. A 640 m tile often contains more than one land use. |
| 2 | The PNGs carry **no metadata at all**: no coordinates, CRS, capture time or sensor. | "Where" and "when" queries are impossible unless metadata travels *with* the tile in the request. This is my first question to you. |
| 3 | Only **7 of EuroSAT's 10 classes** are included. HerbaceousVegetation, Pasture and PermanentCrop were dropped. | Real imagery will contain land the model has never seen. A closed-set classifier will force it into one of the 7 classes, often with high confidence. I need an "unfamiliar" signal that doesn't come from softmax. |
| 4 | The data is **perfectly balanced**: 150 tiles per class for training, 30 per class for eval. | Accuracy here won't match accuracy in the field, where the class mix is skewed. With only 30 tiles per class, an 80% per-class accuracy has a 95% confidence interval of about ±14 pp, so per-class numbers are noisy. Thresholds must not be tuned on the eval set. |
| 5 | Eval filenames are **sorted by class** (`tile_001–030` Forest, `031–060` River, …). | The filename leaks the label. Replaying the eval set in order would look like drift. Shuffle it for any time-based test, and never use the ID as a feature. |
| 6 | There are **no duplicates** between the candidate and eval sets. I checked exact pixel hashes, and perceptual hashes including flipped and rotated versions. | The 210 eval tiles are a fair hold-out for anything *we* train. Public EuroSAT checkpoints, though, were very likely trained on these exact tiles, so their scores here can't be trusted. |
| 7 | Some **SeaLake tiles are featureless grey** (e.g. `SeaLake_16`, `_18`, `_103`, eval `tile_155`). They are indistinguishable from haze, cloud or no-data. | The model will learn "flat grey = water", so a cloudy tile in production will probably be called SeaLake with high confidence. Input-quality checks have to be separate from the classifier. |
| 8 | Classes **overlap visually**. Highway tiles contain fields, rivers and buildings. River tiles contain roads. Forest and SeaLake are both dark and flat. | These are the confusions to expect, and the baseline confirms them. One label per tile is a simplification, so I store the full probability vector, not just the top-1. |
| 9 | A quick **colour/texture baseline** (summary statistics + logistic regression) scores **85%** on eval. Highway recall is 60%; SeaLake↔Forest is the other big confusion. Nearest-neighbour on raw pixels scores only 31%. | This is the floor a real model must beat. **Confidence carries real signal:** keeping only predictions with max-prob ≥ 0.7 keeps 74% of tiles at 94% accuracy, and ≥ 0.9 keeps 50% at 99%. This trade-off between coverage and accuracy is the main tool for handling uncertain predictions. |

---

## 2. The system

Everything runs as **one process on one machine**:

- **API layer (FastAPI).** Endpoints for ingest, query, review and health. A CLI for bulk folder drops calls the same pipeline.
- **Pipeline.** One pure function, `process(tile_bytes, metadata) -> record`. The HTTP handler and the batch CLI both call it, so there is exactly one code path to test.
- **Model runtime (ONNX Runtime, CPU).** A model is a directory, `models/<version>/`, containing `model.onnx` and `manifest.json`. The manifest holds the class list in index order, the preprocessing spec, the weights checksum, a hash of the training data, validation metrics and the threshold policy. Swapping a model means dropping in a new directory; nothing else changes.
- **Storage.** SQLite for records, plus a content-addressed folder for raw tiles.
- **Self-monitoring.** Nobody watches the service live, so it watches itself. It runs a self-test on a few known ("golden") tiles at startup and daily, and keeps a daily health rollup table. Part 3 goes into more detail.

### How a tile flows through it

```
 Tile + optional metadata (HTTP upload, or a folder drop picked up by the CLI)
            │
            ▼
 1. Validate ─────────► reject (400) with a reason: not a PNG, not 64×64, not 3-band, corrupt
            │
            ▼
 2. Hash (sha256) ────► already processed by this model version? return the stored result
            │
            ▼
 3. Quality checks      flags only, never a reject: blank, grey/haze-like, saturated
            │
            ▼
 4. Store raw tile      blobs/<sha256[:2]>/<sha256>.png  (content-addressed, deduplicated)
            │
            ▼
 5. Classify            preprocessing + model from ONE versioned artifact
            │           → 7 class probabilities + embedding vector
            ▼
 6. Score               confidence, margin (top1 − top2), unfamiliarity (embedding
            │           distance to training data) → status via the active policy
            ▼
 7. Persist             one SQLite transaction: tile row + prediction row
            │           (if the model crashes, the tile is saved as status=failed for retry)
            ▼
 JSON response ──► analysts query over REST, or with read-only SQL / CSV export
```

### API surface (v1)

| Endpoint | Purpose |
|---|---|
| `POST /v1/tiles` | Upload a tile plus optional metadata. Returns the tile ID and its prediction. Idempotent on image hash + model version. |
| `GET /v1/tiles/{id}` | One tile's result, with its scores and model version. |
| `GET /v1/predictions?label=&status=&min_confidence=&model_version=&from=&to=` | Filtered, paginated listing. |
| `GET /v1/stats?group_by=label` | Counts and shares per class. |
| `GET /v1/review-queue` / `POST /v1/tiles/{id}/label` | Uncertain tiles for a human to check, and the human's corrected label. |
| `GET /healthz` | Model checksum verified, self-test passed, database writable, disk free. |

---

## 3. What gets stored

The principle: store enough to **reproduce, re-score and audit** any answer without needing the original sender.

| Table | Key columns | Why |
|---|---|---|
| `tiles` | `tile_id` (sha256), `source_name`, `received_at`, optional `captured_at`, `lat/lon` or `bbox` + CRS, `sensor`, `width/height/bands`, `quality_flags`, `blob_path` | One row per unique image. Using the hash as the ID gives de-duplication for free. |
| `models` | `model_version`, `classes` (ordered), `weights_sha256`, `preprocess` spec, `val_metrics`, `created_at` | Every prediction points here. The class order is stored because mixing up index↔label is the classic silent bug. |
| `predictions` | `tile_id`, `model_version`, `label`, `confidence`, `margin`, `probs` (all 7), `ood_score`, `status`, `policy_version`, `latency_ms`, `created_at` | **Append-only.** A new model adds rows instead of overwriting old ones, and a "current" view picks the active model. Keeping the full `probs` allows re-thresholding and top-k answers later. |
| `reviews` | `tile_id`, `true_label`, `reviewer`, `created_at` | Analyst corrections become ongoing ground truth: they measure real-world accuracy and feed retraining. |
| `daily_health` | date, counts, class mix, mean confidence, % needs-review, % unfamiliar, errors, p95 latency | Lets someone arriving a month later see what happened while nobody was watching. |

Embeddings are 512 floats, about 2 KB per tile, and live in a side table. They're optional but cheap, and they enable similarity search and drift checks.

---

## 4. Decisions and trade-offs

### 4.1 Which model

| Option | For | Against |
|---|---|---|
| Public EuroSAT-fine-tuned checkpoint | High accuracy with no training (97.7–100% on tiles they never saw). | Outputs 10 classes, not 7. Measured: 81–91% of our eval tiles were in their training data. Preprocessing is often undocumented, and the best ones declare no licence. |
| Zero-shot CLIP | No training; labels are just text. | Measured 45–71% on our tiles; large and slow on CPU. |
| Frozen pretrained backbone + logistic-regression head | Trains in seconds; up to 97.6% measured; embeddings reusable for unfamiliarity and similarity. | 1–6 points below fine-tuning; weak for ImageNet-1k CNNs (ResNet-18: 92.4%). |
| **Fine-tune a small pretrained transformer (ViT-S/16 @112 px)** ✅ | 98.9% measured; ~25 min on CPU; ~30 ms per tile in ONNX, the same cost as ResNet-18 @224; strong embeddings. | DINOv3 weights carry a custom licence (fallback: Apache-2.0 ViT-S/16, 98.1%). |
| Hand-crafted features + logistic regression | Tiny and fully explainable; measured at 85%. | Low ceiling. |

**Pick: a fine-tuned ViT-S/16 at 112 px, served via ONNX.** My first draft said "frozen ResNet-18 + head". That was an unmeasured default, so I benchmarked the alternatives on this data before committing (`model_selection_thoughtprocess.md`). The top fine-tuned backbones are statistically tied (98.1–98.9%), and at 112 px the ViT is as cheap on CPU as ResNet-18 @224. The tie-breakers:
- best frozen features at that cost;
- clean ONNX export;
- the same family as the multi-sensor Earth-observation foundation models GalaxEye would move to later.

The hand-crafted model stays as the baseline to beat, and a specialized EuroSAT checkpoint serves as a second-opinion monitor. The model sits behind a narrow interface (`image → probs, embedding`, plus the manifest), so swapping weights touches nothing else.

### 4.2 Predictions the model isn't sure about

"Unsure" comes in three kinds, and each needs a different fix:

1. **Ambiguous between known classes**, e.g. Highway vs River. Shows up as low confidence or a small margin. → `needs_review`
2. **Unlike anything seen in training**, e.g. a cloud, a vineyard, a new sensor. Softmax can still be confident here, so I use a separate score: how far the tile's embedding is from its nearest training embeddings, cut off at a high percentile of the distances seen on validation data. → `unfamiliar`
3. **Bad input**, e.g. blank, hazy or saturated. Cheap, deterministic checks run before the model. → quality flag

What could we *do* with uncertain predictions?

- **(a) Always return the top-1 label.** Simple, but at baseline accuracy it is silently wrong about 1 time in 7.
- **(b) Abstain below a threshold.** Safer, but coverage drops and analysts see gaps.
- **(c) Return top-k or all probabilities.** Honest, but harder to consume.
- **(d) Send them to a human review queue.** Best quality, but needs people.

**Pick:** store everything, meaning the full probabilities and every score. The `status` (`confident` / `needs_review` / `unfamiliar` / `rejected`) is derived from a **versioned policy**. Queries return only confident results by default, with a flag to include the rest. Uncertain tiles go to the review queue, and the reviews become ground truth.

The threshold comes from the coverage-vs-accuracy curve on a **validation split carved out of `candidate_tiles`, never the eval set**. Where exactly to set it depends on what errors cost, which is question 4 below. If the model turns out to be badly calibrated, a one-parameter temperature scaling fitted on the same split makes "0.8" mean roughly "right 80% of the time".

### 4.3 What "querying the results" should mean

In increasing order of effort:

1. **Lookup**: the result for tile X, with its model version and scores.
2. **Filter**: e.g. label = SeaLake AND status = confident AND received this week AND model = v2.
3. **Aggregate**: counts or share per class, over time or area. Raw counts of noisy labels are biased: if Highway is often called River, River gets over-counted. The confusion matrix measured on eval can correct for this, so aggregates are reported with that uncertainty attached.
4. **Review queue**: the tiles most worth a human's time to label next.
5. **Spatial / temporal**: tiles inside a polygon, or land-use change between dates. Only possible if tiles carry coordinates and time (finding #2).
6. **Similarity**: "find tiles like this one", using the embeddings.

**Pick:** levels 1–4 for v1, over the small REST API. Analysts who prefer their own tools also get read-only SQL and CSV export. Level 5 is designed for (the geo columns exist but are nullable) and is blocked on metadata. Level 6 comes later.

### 4.4 Storage engine

| Option | For | Against |
|---|---|---|
| **SQLite (WAL mode)** ✅ | Zero-ops single file; trivial to back up or carry off an air-gapped box; fine up to millions of rows. | One writer at a time; no native spatial index. |
| Postgres + PostGIS | Real spatial queries; concurrency. | Another service to install, patch and monitor offline. |
| Parquet + DuckDB | Excellent for analytics. | Awkward for row-by-row transactional writes. |
| One JSON file per tile | Simplest possible. | No real querying. |

**Pick: SQLite, with a schema kept Postgres-compatible.** I'd move to PostGIS once tiles have coordinates and spatial queries matter; SpatiaLite/GeoPackage is the lighter alternative and is SQLite underneath. For heavy analytics, DuckDB can read the SQLite file directly.

### 4.5 Synchronous vs asynchronous ingestion

A 64×64 tile classifies in tens of milliseconds on CPU. So for v1, a **synchronous request → response** is right: it is the simplest to reason about and to debug.

For bulk ingest, such as a whole scene cut into thousands of tiles, I'd add a job table in the same SQLite database and a worker process. I'd choose that over Redis, Kafka or Celery because it means fewer moving parts on isolated hardware. Since the pipeline is one pure function, moving from synchronous to queued changes the wiring, not the logic.

### 4.6 Actually running offline

- **Nothing downloads at runtime.** Model weights ship inside the release and are checksum-verified at startup. Library auto-downloads are switched off (`HF_HUB_OFFLINE=1`, a local `TORCH_HOME`).
- **Dependencies are pinned** (a lockfile plus a wheelhouse, or a `docker save` tarball), so installation works with no internet.
- **ONNX Runtime does the serving:** it is small, CPU-optimised and has no training dependencies. PyTorch lives only on the training machine.
- **Preprocessing ships with the model**: resize method, channel order and normalisation. Any mismatch between training and serving produces confident, wrong answers that nobody notices, and this is the most likely way that happens.
- **Model updates arrive on physical media** as a new `models/<version>/` directory. Old versions are kept, so old results stay explainable.

---

## 5. Assumptions

- Inputs are single 64×64 RGB Sentinel-2 tiles like the sample. In v1 anything else is rejected with a clear error; nothing is silently resized.
- Classification is single-label over a closed set of these 7 classes. "None of these" is handled by the `unfamiliar` status, not by an 8th class.
- One node, modest volume (thousands to low millions of tiles), a handful of analysts, no hard latency SLA.
- The labels in `candidate_tiles` are good enough to train on. The grey SeaLake tiles make me slightly doubt this.
- Location and time metadata *may* arrive with a tile. The schema has room for it but doesn't require it.
- For this exercise, the isolated environment takes care of security (auth, encryption at rest). In reality I'd confirm that.

## 6. Questions I'd ask you

1. **What arrives with a tile?** Coordinates, CRS, capture time, scene ID, sensor? This decides whether spatial and change-over-time queries are possible at all.
2. **Will inputs stay 64×64 RGB?** Or will we get full scenes that need tiling, 13-band multispectral data, or SAR from GalaxEye's own sensors? Any of these changes the model interface and the validation.
3. **Are the 7 classes fixed?** What should happen to land that is none of them (pasture, orchards) or to cloud?
4. **Which mistakes are expensive?** Missing water and falsely flagging "industrial" cost different amounts. That sets the thresholds, possibly per class.
5. **How much traffic, and in what shape?** Tiles per day, bursts, batch drops or a continuous stream?
6. **What is the target hardware?** CPU cores, RAM, disk, OS. Is Docker allowed?
7. **Who queries, and how?** Through an API, SQL, QGIS, a dashboard? Which export formats?
8. **Is anyone available to review uncertain tiles?** And how do new models get onto the box?
9. **What are the retention rules?** Can we keep raw tiles, for how long, and is the imagery sensitive?
10. **What happens to a re-sent tile?** Return the stored result, or re-run it?
11. **Is the balanced 30-per-class eval set representative?** If not, what class mix should we expect in production?

## 7. What Part 2 will cover

- **Build:** `POST /v1/tiles` → validate → hash → store the raw tile → classify (fine-tuned ViT-S/16 @112 px, ONNX) → store the prediction in SQLite → return JSON. Also `GET /v1/tiles/{id}`, a CPU fine-tuning + ONNX export script, and an evaluation script over `eval_set` that reports accuracy, the confusion matrix and the coverage-vs-accuracy curve.
- **Stub or skip, and say so:** the review queue UI, the async job worker, spatial queries, auth and the drift dashboard.

---

## Appendix A: How I analysed the data

All checks ran locally with Pillow, NumPy, imagehash and scikit-learn; the scripts are in `experiments/data_analysis/` (`data_profile.py`, `dups.py`, `baseline.py`, `montage.py`).

- **Inventory.** 7 classes × 150 candidate tiles, plus 210 eval tiles (30 per class). Every file is a 64×64 RGB PNG with no embedded metadata, between 2.6 and 10.2 KB (median 6.6 KB). Pixel values range from 12 to 255; there are no zeros, so no no-data padding.
- **Leakage.** No exact pixel duplicates among all 1,260 tiles. No eval tile lies within perceptual-hash distance ≤ 6 of any candidate tile, including flipped and rotated versions.
- **Per-class colour.** Forest and SeaLake are dark with very little variation inside a tile (per-channel std ≈ 2–5). Industrial is the brightest and most textured class. AnnualCrop has the widest colour spread (bare soil vs green fields). Candidate and eval statistics match closely, so there is no obvious shift between the two splits.
- **Odd tiles.** 9 grey tiles with almost no texture, all labelled SeaLake: 7 in the candidates and 2 in eval.
- **Baselines.** These were exploratory: trained on the candidates and scored on eval.

  | Model | Eval accuracy | Notes |
  |---|---|---|
  | 1-NN on raw pixels | 31% | Raw pixels are useless as features. |
  | Random forest on colour/texture stats | 83% | Highway recall 50%, River 60%. |
  | Logistic regression on colour/texture stats | 85% | Highway recall 60%; 6 SeaLake tiles predicted as Forest. |

  For the logistic regression, confidence threshold vs coverage:

  | max-prob ≥ | Tiles kept | Accuracy on kept tiles |
  |---|---|---|
  | 0.5 | 91% | 87% |
  | 0.7 | 74% | 94% |
  | 0.9 | 50% | 99% |

  Mean confidence was 0.86 on correct predictions and 0.63 on wrong ones. I read these thresholds off the eval set only to understand the data. In Part 2 they will be chosen on a validation split, so the eval set stays untouched.

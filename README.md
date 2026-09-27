# Satellite image classification system

GalaxEye take-home (Backend Engineer, ML Systems). The service is an offline land-use classifier for 64×64 Sentinel-2 RGB tiles: it ingests a tile, classifies it with a locally-run model, stores the result, and lets an analyst query everything, entirely on CPU with no network.

## Live demo

**Open the app: [https://confidential-personalized-reached-brad.trycloudflare.com/?key=2PEcAFu3RuhKtPKUO8FnbLv-AvV331o9](https://confidential-personalized-reached-brad.trycloudflare.com/?key=2PEcAFu3RuhKtPKUO8FnbLv-AvV331o9)**

This link opens the classify page. The other pages:
- [Query results](https://confidential-personalized-reached-brad.trycloudflare.com/explore): lookup, filter, aggregate, review queue, CSV export;
- [Health dashboard](https://confidential-personalized-reached-brad.trycloudflare.com/health);
- [API docs](https://confidential-personalized-reached-brad.trycloudflare.com/docs).

The key in the link opens only this classifier. It's stored in a cookie on the first visit, so the other links work without it.

> The demo runs on a temporary cloud GPU instance, reached through a Cloudflare quick tunnel. The tunnel address changes whenever the instance restarts, and the link stops working once the instance is shut down. To run it yourself, see [Run it](#run-it).

## Where each part of the assignment is answered

| Part | Where |
|---|---|
| **Part 1: Your approach** | [`DESIGN_NOTE.md`](DESIGN_NOTE.md), the design note: data findings, architecture and tile flow, storage, decisions and trade-offs, assumptions, questions. The model choice is covered in [`model_selection_summary.md`](model_selection_summary.md) (two pages) and [`model_selection_thoughtprocess.md`](model_selection_thoughtprocess.md) (the full evidence: 58 model configurations benchmarked on CPU). The code and raw results behind those numbers are in [`experiments/`](experiments/). |
| **Part 2: A working slice** | This README: what's built, how to run it, and the code structure below. |
| **Part 3: Problem-solving** | [The last section of this README](#part-3--problem-solving). |

## Part 2 in short

**The core path:** `POST /v1/tiles` → validate → hash (idempotent) → quality flags → store the raw tile → classify with **DINOv3 ViT-S/16, fine-tuned at 112 px** and served via ONNX Runtime → score (calibrated confidence, "unfamiliar" distance, status) → persist in SQLite → JSON.

**On top of it:**
- **querying the results** (design note §4.3, levels 1–4) in the API and the `/explore` page;
- **a health dashboard;**
- **an optional access key** for sharing;
- **a verified offline mode.**

What's deliberately stubbed or skipped is listed in [What is implemented, what is stubbed](#what-is-implemented-what-is-stubbed).

**Code structure: which folder holds what**

| Folder | Purpose |
|---|---|
| `tile_service/` | **The service that ships.** Runtime only: FastAPI + ONNX Runtime + NumPy + Pillow + SQLite, no PyTorch. It contains:<ul><li>the API (`api.py`);</li><li>the tile pipeline (`pipeline.py`);</li><li>input checks (`validation.py`);</li><li>model loading and verification (`model.py`);</li><li>scoring and status policy (`scoring.py`);</li><li>storage and queries (`storage.py`);</li><li>corrected aggregates (`aggregate.py`);</li><li>the access key (`auth.py`);</li><li>the three browser pages in `static/ui/`.</li></ul> |
| `training/` | Development-machine code that produces the model: `train.py` (CPU fine-tune → calibration → thresholds → ONNX artifact) and `evaluate.py` (sends the eval set through the running API → `reports/eval_report.md`). |
| `models/` | The trained, versioned model artifact the service loads (`model.onnx` + `manifest.json` + reference embeddings + self-test tiles), with `models/active` pointing at the current one. |
| `weights/` | Vendored pretrained backbones (original upstream files + licences), so retraining works offline too. |
| `tests/` | The pytest suite: 44 tests covering the pipeline, API, querying, health, access key, the real model artifact, and offline guarantees. |
| `scripts/` | `fetch_offline_assets.py` (the only step that needs internet; already done) and `offline_check.py` (proves the service and retraining run with the network blocked). |
| `experiments/` | Part 1 evidence: the data analysis and the model-selection benchmark, with the raw results. |
| `reports/` | Training log, evaluation report, offline-check report. |
| `wheelhouse/serve/` | Python wheels for installing the service on an air-gapped Linux x86_64 machine. |
| `be-mlsys-assignment-dataset/` | The provided tiles (EuroSAT subset: `candidate_tiles/` for training, `eval_set/` + labels for evaluation). |

## Key results

The core path from `DESIGN_NOTE.md`, working end to end, is offline and CPU-only at runtime. The model is the one chosen in `model_selection_thoughtprocess.md`.

| Result (from `reports/eval_report.md`, 210 eval tiles sent through the running HTTP server) | |
|---|---|
| Accuracy, all tiles | **98.6%** (207/210; 95% CI 95.9–99.5%) |
| Tiles marked `confident` / accuracy on them | **96.2% / 99.5%** (1 error in 202) |
| `needs_review` (quality-flagged) / `unfamiliar` (far from training data) | 2 tiles / 6 tiles (2 of the 3 errors land here) |
| Model latency per tile / full HTTP request, p50 | ~25 ms / ~34 ms (4 CPU threads) |
| Training time (CPU only, 32 threads) | 8.6 min; validation accuracy 99.4% |
| Offline verification (`reports/offline_check.md`) | **14/14 checks pass**: the service and retraining run with the network blocked and empty download caches |

## Layout (file by file)

```
tile_service/         runtime package (what ships)
  api.py              FastAPI app: POST /v1/tiles, lookup / filter / CSV / stats / review queue / labels, /healthz, pages
  pipeline.py         the tile flow as one function: validate -> hash -> idempotency -> flags -> store -> classify -> score -> persist
  validation.py       strict input checks (reject) + quality flags (never reject)
  model.py            loads models/<version>/, verifies checksums, runs the golden self-test, ONNX inference
  scoring.py          calibrated probabilities, margin, "unfamiliar" score, status policy
  aggregate.py        class shares corrected for the model's measured confusions (+ bootstrap 95% interval)
  storage.py          SQLite schema (tiles / models / append-only predictions) + content-addressed blob store
  auth.py             optional access key for a shareable link (off unless TILE_ACCESS_KEY is set)
  static/ui/          browser pages: / (classify), /explore (query stored results), /health (health dashboard)
training/
  train.py            CPU fine-tune -> calibration -> thresholds -> ONNX export -> manifest -> models/active
  evaluate.py         sends eval_set through the API; writes reports/eval_report.md
  backbones.py        the two pretrained backbones and where their vendored weights live
tests/                44 tests: unit + API (fake model), querying, health, access key, real-artifact and offline checks
scripts/
  fetch_offline_assets.py   the ONLY online step: fetches backbone weights + Swagger UI assets (already done)
  offline_check.py          proves offline operation under a network guard -> reports/offline_check.md
  netguard/sitecustomize.py the guard: blocks + logs any non-localhost DNS lookup / connection (verification only)
models/<version>/     the deployable artifact: model.onnx, manifest.json, reference_embeddings.npy, selftest/
weights/              vendored pretrained backbones (original upstream safetensors + licence files + sha256 manifest)
wheelhouse/serve/     wheels for requirements-serve.txt (Linux x86_64, CPython 3.11) for offline installs
tile_service/static/  vendored Swagger UI 5.33.0, so /docs works without a CDN
reports/              train_log.txt, eval_report.md (+ .json), offline_check.md, server_log.txt, eval_run1_before_blank_fix.md
data/                 runtime state, created on first run: tiles.db + blobs/ (git-ignored)
```

## Run it

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu   # training only
pip install -r requirements-train.txt          # a serving-only box needs just: pip install -r requirements-serve.txt

python training/train.py                       # ~9 min on 32 CPU threads, fully offline (reads weights/); writes models/<version>/
                                               # and points models/active at it. Optional: a trained model is already in models/.
uvicorn tile_service.api:app --host 127.0.0.1 --port 8000
```

Classify a tile (metadata fields are optional):

```bash
curl -F file=@be-mlsys-assignment-dataset/eval_set/tile_197.png -F lat=48.13 -F lon=11.58 -F sensor=S2 \
     -F captured_at=2026-06-01T10:31:00Z http://127.0.0.1:8000/v1/tiles
```

```json
{"tile_id": "f2b34ff5…", "model_version": "dinov3_vits16-112px-20260926-182432", "policy_version": "p1",
 "status": "confident", "label": "Highway", "confidence": 0.9968, "margin": 0.9961,
 "probabilities": {"AnnualCrop": 0.00049, "Forest": 0.00047, "Highway": 0.99677, "…": "…"},
 "ood_score": 0.0267, "quality_flags": [], "latency_ms": 28.5, "lat": 48.13, "lon": 11.58, "sensor": "S2",
 "captured_at": "2026-06-01T10:31:00Z", "idempotent_hit": false, "…": "…"}
```

**Status codes:**
- `201` classified and stored;
- `200` the same bytes were already classified by this model (`idempotent_hit: true`, nothing re-run);
- `400` not a valid 64×64 RGB PNG (the reason is in `detail`);
- `422` bad form field (e.g. `lat` out of range);
- `500` stored, but inference failed (`status: failed`; retried on the next upload);
- `503` model not loaded (see `/healthz`).

Querying the stored results (DESIGN_NOTE.md §4.3). The browser page **`/explore`** covers all of these, with tabs for each level:

| Level | Endpoint | What it gives you |
|---|---|---|
| 1 Lookup | `GET /v1/tiles/{tile_id}`, `GET /v1/tiles/{tile_id}/image` | the tile, every prediction ever made for it (all model versions, failures) and its analyst reviews; the stored PNG |
| 2 Filter | `GET /v1/predictions?label=&status=&min_confidence=&max_confidence=&received_from=&received_to=&captured_from=&captured_to=&bbox=&flag=&reviewed=&q=&sort=&limit=&offset=` | filtered, sorted, paged listing; the total match count is in `X-Total-Count` |
| Export | `GET /v1/predictions.csv?…same filters…` | every matching row as CSV (all 7 probabilities, flags, latest review) |
| 3 Aggregate | `GET /v1/stats?…same filters…` | per-class counts, raw **and corrected** for the model's measured confusions with a 95% interval; counts by status and day; analyst agreement |
| 4 Review queue | `GET /v1/review-queue?…filters…` and `POST /v1/tiles/{tile_id}/label` `{"label", "note", "reviewer"}` | unreviewed `unfamiliar` / `needs_review` tiles, most useful first, each with its reason; labels (the 7 classes, `Other`, `Unusable`) are stored append-only |

Plus:
- `GET /v1/meta`: classes, flags, policy thresholds.
- `GET /health`: the **visual health dashboard**, which refreshes every 30 s and works offline:
  - an overall status and the individual checks, each shown as icon + label against its stated threshold;
  - tiles classified per day by status;
  - the unfamiliar-input rate per day against its alert lines (the drift signal);
  - the recent predicted-class mix against the training mix;
  - latency, disk, and review backlog;
  - the model's golden self-test tiles.

  Its data comes from `GET /v1/ops`.
- `GET /healthz`: the same essentials as compact JSON for monitors (model version, self-test, DB, disk).
- `GET /docs`: OpenAPI.

**About the corrected counts:** `training/evaluate.py` writes `models/<version>/aggregate_calibration.json`, which holds the eval confusion matrix. The stats endpoint uses it to undo the classifier's known mix-ups by inverting P(predicted | true), which stays valid when the real class mix differs from the balanced eval set. With 30 labelled tiles per class the intervals are honest but wide; more labelled data narrows them.

**Read-only SQL for your own tools:**
```bash
sqlite3 "file:data/tiles.db?mode=ro"
```
Useful objects: the views `current_predictions` and `current_reviews`, and the tables `tiles`, `predictions`, `reviews` and `models`.

Configuration comes from environment variables:
- `TILE_MODEL_DIR` (default `models/active`);
- `TILE_DATA_DIR` (default `data/`);
- `TILE_ORT_THREADS` (default 4);
- `TILE_MAX_UPLOAD_BYTES` (default 1 MB).

Evaluate, then test:

```bash
python training/evaluate.py --url http://127.0.0.1:8000    # or without --url: runs the app in-process
python -m pytest                                          # 44 tests, ~5 s
```

## Running offline (nothing needs the network)

Everything the service and the training script need is already in the repository:

| What | Where | How it got there |
|---|---|---|
| Trained model the service runs | `models/<version>/` (`model.onnx` + manifest + reference embeddings + self-test tiles) | `training/train.py` |
| Pretrained backbones for retraining | `weights/<timm_name>/model.safetensors` (+ upstream `LICENSE`, sha256 in `weights/manifest.json`) | `scripts/fetch_offline_assets.py` |
| Swagger UI for `/docs` (FastAPI's default loads it from a CDN) | `tile_service/static/swagger-ui/` | `scripts/fetch_offline_assets.py` |
| Python packages for the service | `wheelhouse/serve/` | `pip download -r requirements-serve.txt -d wheelhouse/serve` |

- **Guards built into the code:**
  - The service imports no ML framework or HTTP client (a test enforces this). It reads only local files and serves `/docs` from local assets.
  - `train.py` sets `HF_HUB_OFFLINE=1` before importing timm and loads the backbone from `weights/`. If that folder is missing, it stops with instructions rather than downloading.
- **Install on an air-gapped Linux x86_64 box:**
  ```bash
  python3.11 -m venv .venv && . .venv/bin/activate
  pip install --no-index --find-links wheelhouse/serve -r requirements-serve.txt
  ```
- **Proof:**
  ```bash
  python scripts/offline_check.py       # ~1 min; add --skip-train to skip the retraining step
  ```
  Every process runs under a network guard that blocks and logs any non-localhost DNS lookup, connection or UDP send, with empty Hugging Face and torch caches. The script:
  1. installs the serving dependencies into a fresh environment from the wheelhouse with `--no-index`; that environment has no PyTorch;
  2. starts the server from that environment and classifies one eval tile per class;
  3. checks idempotency, the read endpoints and `/docs`;
  4. lists the server's sockets (localhost only);
  5. retrains for one epoch from the vendored weights.

  Pass criterion: everything works and the guard logs zero network attempts. Result: `reports/offline_check.md`.

## Sharing a link (access key)

By default the service has no authentication of its own: it binds to 127.0.0.1, and on this instance it is reached through the Caddy edge, which demands the instance's master token. That token also unlocks Jupyter and the other portal apps, so it must not be shared. For a link you *can* share, the service has its own **optional access key**:

- **Turning it on:** set `TILE_ACCESS_KEY`, or `TILE_ACCESS_KEY_FILE=<file>`, which keeps the key out of process listings. With it set, **every** route (page, API, `/docs`, static files, `/healthz`) needs the key.
- **Ways to present the key:**
  - `?key=<key>` in the link: on a GET the key is moved into a `Secure; HttpOnly; SameSite=Lax` cookie and stripped from the URL by a redirect;
  - an `X-API-Key: <key>` header;
  - `Authorization: Bearer <key>`.
- **Without the key:** browsers get a small key-entry page (401); scripts get a JSON 401.
- **Scripts:**
  ```bash
  curl -H "X-API-Key: $KEY" -F file=@tile.png <url>/v1/tiles
  python training/evaluate.py --url <url> --key "$KEY"
  ```

**How it's wired on this Vast instance:**
- **Service:** the supervisor service `tile-classifier` (`/opt/supervisor-scripts/tile-classifier.sh`) reads the key from `.secrets/tile_access_key` (mode 600, git-ignored).
- **Shareable link:** a Cloudflare quick tunnel points **straight at the app** (`http://localhost:18000`), bypassing Caddy. The only credential on that link is the classifier's own key, which grants nothing else on the machine.
- **Owner route:** the Caddy route (public port 19900) still demands the master token, and now the key as well.

Commands:
```bash
# (re)create the shareable tunnel - quick-tunnel URLs change whenever the instance restarts
curl -s -X POST http://localhost:11111/start-quick-tunnel/http%3A%2F%2Flocalhost%3A18000
curl -s http://localhost:11111/get-all-quick-tunnels          # the entry for http://localhost:18000
echo "https://<tunnel-host>/?key=$(cat .secrets/tile_access_key)"

# rotate the key: every previously shared link and cookie stops working immediately
( umask 077; python3 -c "import secrets; print(secrets.token_urlsafe(24))" > .secrets/tile_access_key ) && supervisorctl restart tile-classifier

# stop sharing entirely
curl -s -X POST http://localhost:11111/stop-quick-tunnel/http%3A%2F%2Flocalhost%3A18000
```

Anyone with the key can upload tiles, and each one is stored (~7 KB). There's no rate limiting in this slice, so rotate the key or stop the tunnel once the link has done its job.

## What is implemented, what is stubbed

| Design-note piece | In this slice |
|---|---|
| `POST /v1/tiles`: validate → hash → idempotency → quality flags → store raw tile → classify → score → persist in one transaction | **Implemented** |
| Model as a versioned artifact (ONNX + manifest with class order, preprocessing, checksums, data hash, calibration, policy) | **Implemented** |
| Startup checks: SHA-256 of the artifact files + golden self-test tile per class; degrade to 503 instead of crashing | **Implemented** |
| Statuses `confident` / `needs_review` / `unfamiliar` / `failed`; all 7 probabilities and scores stored | **Implemented** |
| Append-only predictions + `current_predictions` view; a failed inference is kept and retried | **Implemented** |
| Querying levels 1–4: lookup, filter, aggregate with confusion-matrix correction, review queue + analyst labels; CSV export; read-only SQL | **Implemented** (API + `/explore` page) |
| Browser pages `/` and `/explore` (self-contained, offline, phone-friendly) and optional access key for sharing | **Implemented** |
| Level 5, spatial / temporal | **Partly**: bounding-box and capture/received date filters work on the existing columns. Polygon search and land-use change between dates are **skipped**: the sample tiles carry no coordinates or repeat captures |
| Level 6, similarity search ("tiles like this one") | **Skipped**, as the design note's pick says: embeddings are computed but not stored yet |
| Rate limiting, per-user keys | **Skipped**: a single shared key only |
| Async job queue / folder-drop batch ingest | **Skipped**: requests are synchronous; `TilePipeline.process()` is the reusable unit a worker would call |
| Re-scoring stored results under a new policy version | **Skipped**: probabilities and scores are stored so it's possible, but not built |
| `daily_health` rollups, drift monitoring, auth, TLS, retention | **Skipped** |

## Design decisions as they appear in code

- **Preprocessing lives inside the ONNX graph.** The graph takes raw 0–255 pixels and does ÷255, bicubic 64→112 and normalisation itself.
  - The model-selection experiments found a downloaded model that drops from 99% to 13% under the wrong preprocessing, silently.
  - Here the service can't get preprocessing wrong. Export parity with PyTorch is checked at training time (max |Δlogit| 4.5e-5, 100% argmax agreement) and recorded in the manifest.
- **Every threshold comes from the manifest, chosen on validation data only.** That covers the temperature (0.58), `min_confidence` (0.5) and `ood_threshold` (0.188, the 99th percentile of validation distances). `eval_set` is used only by `evaluate.py`.
- **Three kinds of "unsure" are kept apart:**
  - calibrated confidence, for ambiguity between known classes;
  - embedding distance to the training tiles, for "unlike anything seen";
  - pixel-level quality flags.

  The evaluation shows why all three are needed. A uniform grey no-data tile gets a *99%-confident SeaLake* from the model, and its embedding distance is low too; only the `blank` flag catches it. Noise and checkerboard tiles are caught by the embedding distance.
- **Identity = SHA-256 of the uploaded bytes.** That's the idempotency key and the blob address.

## Honest notes and known limitations

- **I fixed a quality-rule bug after the first eval run.** The kept record is `reports/eval_run1_before_blank_fix.md`.
  - The original `blank` rule (texture < 1.0) sent 9 correctly-classified calm-water tiles to review.
  - I checked the *training* tiles, not the eval set: the rule also fired on 37 of the 150 SeaLake training tiles. Deep water is legitimately that flat.
  - `blank` now means a constant fill-value tile (value range ≤ 2 in every band). No real training tile comes close; the minimum is 6.
  - The model and all thresholds were unchanged. Accuracy is the same; only the review load changed, from 11 tiles to 2.
- **The confidence threshold sits at its floor (0.5).** The 158 validation tiles are 99.4% correct at full coverage, so no higher threshold was justified by the data. The right value depends on what errors cost GalaxEye (design-note question 4) and on a larger validation set. The policy is versioned so it can change without retraining.
- **One confident error:** `tile_034` (River → Highway, 0.98). A calibrated 0.98 still fails about 1 time in 50.
- **Concurrency:** two *simultaneous* uploads of the same new tile can both run the model and add two prediction rows. That's harmless (append-only; the latest wins), but it isn't a strict once-only guarantee. SQLite serialises writers, which is fine for one node.
- **Identity by bytes:** the same pixels re-encoded differently get a new id. Hashing decoded pixels would fix that, at the cost of storing a second hash.
- **Licence:** the DINOv3 weights carry Meta's custom licence. `python training/train.py --backbone vit_s16_in21k` produces the Apache-2.0 fallback (98.1% in the experiments) with no code changes.
- **The offline guard is Python-level.** It patches `socket`, so a C extension opening its own sockets would bypass it. The check also lists the server's actual sockets with `ss` (only 127.0.0.1). ONNX Runtime, NumPy and Pillow don't open sockets. A kernel-level test (network namespace) isn't possible in this unprivileged container.
- **The wheelhouse is platform-specific** (Linux x86_64, CPython 3.11) and covers the *serving* dependencies only. Training dependencies (including the ~190 MB CPU PyTorch) are installed in `.venv` here. For another offline box: `pip download -r requirements-train.txt -d wheelhouse/train` on a connected machine of the same platform.

## Part 3 · Problem-solving

### 1. The classifier turns out to be wrong about 30% of the time. What do you do, and how do you decide whether it's "good enough"?

**Take the 30% apart before acting on it.** One error rate hides almost everything that matters.
- **Is the number solid?** Ask four things:
  - How many labelled examples is it based on, and what's the confidence interval?
  - Is it a random sample of real traffic, or a hand-picked or hard-case set?
  - Did the labellers follow one written definition of each class? On land use, two people can disagree on 10–20% of tiles, which puts a ceiling on any model.
  - Is it measured on data like what the model will see in production?
- **Where are the errors?** Use a per-class confusion matrix, error rates by input condition (cloud, season, region, sensor, processing level), and error against confidence. There are usually three patterns, and each points to a different fix:
  - **concentrated** in a few classes that genuinely look alike at this resolution;
  - **systematic** in one condition: domain shift, or a pipeline bug such as preprocessing that differs from training;
  - **open-set:** inputs that belong to none of the classes but are forced into one.
- **Is it the model or the pipeline?** Re-run a sample offline and compare the input statistics with the training data. A sudden jump to 30% is more often a data or pipeline change than a bad model.

**"Good enough" is a decision about use, not a threshold on accuracy.**
- **Name the decision the output feeds, and what each kind of error costs.** A missed flood and a false alarm are not equally bad. Accuracy weighs them the same, so judge by per-class precision/recall, or by expected cost.
- **Compare against the real alternative,** not against perfection. That might be people labelling everything (with its cost and delay), a simple baseline (always predicting the majority class, a vegetation-index rule, last year's map), or no information at all. 70% can be excellent when the alternative is 40%, and useless when the task needs 99%.
- **Write the acceptance test down before measuring,** on data that represents deployment. For example: "≥95% precision on the tiles it auto-accepts, at ≥50% coverage, on ≥500 randomly sampled labelled tiles".

**A model that's 30% wrong overall can still be useful,** because its errors are rarely spread evenly:
- **Triage (selective prediction):** automatically accept the predictions it's confident about, and send the rest to people. The coverage-vs-accuracy curve shows whether this works. If its most confident 60% are 97% right, it removes 60% of the manual work.
- **Totals rather than individual tiles:** if the error rates are measured, class counts can be statistically corrected for them. Area estimates stay unbiased even when many single tiles are wrong.
- **Partial scope:** use it only for the classes or conditions where it's reliable, and abstain elsewhere.
- **Support rather than decision:** ranking tiles for human attention tolerates far more error than taking automatic action.

**Then improve it, cheapest fix first:**
1. fix data or pipeline problems;
2. get labels from the deployment domain (the people reviewing uncertain predictions produce exactly these);
3. retrain or fine-tune on them;
4. revisit the class definitions: merge classes that can't be told apart at this resolution, and add an "other/unknown" class;
5. add information: more spectral bands, SAR, several dates, or more surrounding context;
6. only then try a bigger model.

*In this project the design is built for that loop:*
- the versioned confidence / unfamiliar policy is the triage;
- the review queue collects target-domain labels;
- `/explore → Aggregate` reports confusion-corrected totals with intervals;
- fine-tuning takes minutes on CPU, so retraining on the new labels is cheap.

### 2. The service runs offline with no one watching. A month after deployment, how would you know it's still working?

Nobody watching means the service has to check itself and keep the evidence. "Working" has three layers, and each needs its own signal:

1. **It runs** (no labels needed). Supervisor restarts it on crash. At startup it re-verifies the model files' checksums and re-classifies its golden self-test tiles, which catches a corrupt file, a changed runtime, or broken preprocessing. `/health` shows that, plus failures, latency, disk, and database status.
2. **Its inputs still look like its training data** (no labels needed; the earliest warning). Signals to watch:
   - **the unfamiliar-input rate:** about 1% by construction, with alert lines at 5% and 15% on `/health`;
   - **the predicted-class mix** against its usual mix (also on `/health`);
   - **the needs-review rate, confidence distribution, and quality-flag rates;**
   - **per-channel input statistics** against the training tiles (not built yet).

   A jump in any of these on a particular date usually means something upstream changed.
3. **Its answers are still right** (needs labels). Route a small **random audit sample**, say 1% of *confident* tiles, into the review queue alongside the flagged ones. That gives an unbiased monthly accuracy estimate with an interval. The analyst-agreement number on `/explore` alone is biased toward hard cases. A second model, such as the downloaded EuroSAT checkpoint, can run beside it as a disagreement alarm.

The hard part offline is **getting the signal to a person.** A dashboard only helps if someone opens it. So:
- write a daily health summary to disk and keep those rollups, so a visitor a month later sees the history, not just today;
- expose the status to the host's own monitoring (`/healthz` returns 503 when degraded);
- piggyback the report on whatever leaves the box: a local mail relay, or the physical-media sync that brings model updates in.

**Built today:** startup self-test, `/healthz`, and the `/health` dashboard with checks and drift charts. **Not built yet:** periodic self-tests, audit sampling, persisted daily rollups, input-statistics drift, and a push channel.

### 3. Tiles are coming in fine, but the stored results look wrong. Your actual steps, in order.

The principle: cheap checks that split the problem in half come first; data before model; reproduce before theorising.

1. **Make "wrong" concrete.** Get 5–10 specific tile IDs and what they should be. Establish since when it has been happening and how we know: an analyst report, or a shift in the class mix on `/health`.
2. **Rule out the reading path.** Query SQLite read-only for those IDs and check that the API and UI show exactly what's stored, and that each stored label is the argmax of its stored probabilities. That rules out a display or mapping bug in minutes.
3. **Check what produced them.** Look at the `model_version` and `policy_version` on those rows against when "wrong" started. Did `models/active` change? Were there restarts or errors in the logs? Did the self-test pass at that time?
4. **Reproduce offline.** Take the stored raw PNG (it's content-addressed, so re-hash it and check it matches the tile ID) and run it through the same model artifact with the same code; the pipeline is one function.
   - **Same wrong answer:** the model is doing what it's told, so the problem is in the inputs. Go to step 5.
   - **Different answer:** something differed at serve time: model files, runtime or library versions, or thread-count effects. Compare manifests, checksums and pinned versions.
5. **Look at the inputs.** View the tiles, and compare their per-channel statistics with the training tiles. Look for an upstream change: RGB vs BGR channel order, different bit depth or contrast stretch, resampling from another resolution, a different product level, a cloudy season. The unfamiliar-rate and flag charts on `/health` usually show the date it happened.
6. **Check the label contract.** Confirm the class order in the manifest matches how labels are written, and that the policy thresholds are what they should be. A wrong threshold changes statuses, not labels.
7. **Fix it at the right layer and prevent a repeat:**
   - add a regression test for that failure mode (a golden tile, or an input-statistics check);
   - re-score the affected tiles by *appending* predictions under the fixed version, so the history stays;
   - tell whoever consumed the wrong results (CSV exports, dashboards).

### 4. What's the weakest part of your design, and what would break it first?

**The weakest part is the assumption that production tiles look like EuroSAT:** 64×64, 8-bit RGB Sentinel-2 rendered with EuroSAT's contrast stretch, from seven classes. Everything is fitted to that:
- the model and its temperature;
- the thresholds, chosen on only 158 validation tiles (the confidence threshold sits at its floor because that data can't justify more);
- the reference embeddings behind the unfamiliar score.

When inputs violate the assumption, the safety net is a single distance threshold, and softmax confidence stays high on things the model has never seen. The evaluation showed this directly: a uniform grey no-data tile got **99% SeaLake**, and only the pixel-level quality flag caught it.

**What breaks it first: a quiet change upstream in how tiles are made.** For example, a different contrast stretch, L2A instead of L1C, or a new tiler. The tiles are still valid 64×64 RGB PNGs, so validation passes. The model returns confident labels, and they're stored as `confident`. If the shift is large, the unfamiliar rate climbs on `/health`, but nobody is watching. If it's subtle, the embeddings may never cross the threshold at all.

**How I'd harden it, in order:**
1. Make the input contract explicit: the tile carries sensor, processing level and stretch in its metadata, and unknown values are rejected.
2. Add an input-statistics drift check against the training distribution.
3. Add the random audit sample, for a real accuracy number.
4. Push health reports instead of waiting to be read.
5. Retrain on GalaxEye's own imagery as soon as labels exist.

**The next weakness is operational:** one process, synchronous inference, and a single SQLite writer. That's fine at the ~70–100 tiles/s measured on 4 threads, but a burst of many whole scenes would queue behind HTTP; the answer is the job table plus a worker from the design note. There's also no retention or backup policy yet, so the disk fills eventually; `/health` warns below 1 GB.

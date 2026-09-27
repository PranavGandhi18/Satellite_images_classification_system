# Offline check

Run 2026-09-26 19:15:26 UTC by `python scripts/offline_check.py`. Every Python process ran with `scripts/netguard/sitecustomize.py` (blocks and logs any non-localhost DNS lookup, connection or UDP send; allowed localhost connections are logged too), `HF_HUB_OFFLINE=1`, no proxy variables, and empty Hugging Face / torch caches.

**Result: ALL CHECKS PASSED** (14/14)

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | guard blocks and logs outbound traffic (DNS lookup + raw IP connect) | PASS | logged: ['blocked_dns', 'blocked_connect'] |
| 2 | install serving deps from wheelhouse/serve with --no-index | PASS |  |
| 3 | that environment has no PyTorch / timm installed at all | PASS |  |
| 4 | service starts offline; artifact checksums + self-test pass | PASS | model dinov3_vits16-112px-20260926-182432 |
| 5 | classify one eval tile per class (POST /v1/tiles -> 201) | PASS | 7/7 correct; Forest->Forest (confident, 33 ms), River->River (confident, 26 ms), Residential->Residential (confident, 26 ms), Industrial->Industrial (confident, 25 ms), AnnualCrop->AnnualCrop (confident, 25 ms), SeaLake->SeaLake (needs_review, 25 ms), Highway->Highway (confident, 25 ms) |
| 6 | re-post is an idempotent hit (200) | PASS |  |
| 7 | stored result reads back (GET /v1/tiles, /v1/predictions) | PASS |  |
| 8 | /docs page references no external URL and its assets are served locally | PASS | external URLs: none |
| 9 | all of the server's sockets are on localhost (ss -tuanp) | PASS | 2 sockets: ['0.0.0.0:*', '127.0.0.1:37292', '127.0.0.1:8765'] |
| 10 | guard was active in the server process | PASS |  |
| 11 | server made zero network attempts (none blocked, no local/proxy connects) | PASS | [] |
| 12 | offline retraining (1 epoch) from vendored weights succeeds | PASS | 45s; pretrained sha256 matches weights/manifest.json: True |
| 13 | training made zero network attempts (guard active, none blocked, no local connects) | PASS | [] |
| 14 | empty Hugging Face / torch caches stayed empty | PASS |  |

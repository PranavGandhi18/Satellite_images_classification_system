# Experiments

Code and raw results behind the numbers in `DESIGN_NOTE.md` and `model_selection_thoughtprocess.md`.

| Folder | What it backs |
|---|---|
| `data_analysis/` | `DESIGN_NOTE.md` §1 and Appendix A: dataset profile, leakage/duplicate checks, colour-texture baseline, image montages |
| `model_selection/` | Every table in `model_selection_thoughtprocess.md`: 31 frozen-backbone probes, 8 CPU fine-tunes, 19 specialized checkpoints, leakage matching, ONNX latency |

## Setup

```bash
python3.11 -m venv .venv && . .venv/bin/activate     # or: uv venv
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r experiments/requirements.txt
```

**Network.** The *experiments* download pretrained weights (Hugging Face hub / torchgeo, a few GB in total) and, for the leakage check, ~280 MB of public EuroSAT copies. They cache in `HF_HOME` / `TORCH_HOME`. The *service* itself is meant to run offline with vendored weights, so this is a one-time, online, development-machine step.

**Data location.** The scripts find the dataset at `../../be-mlsys-assignment-dataset` relative to themselves. Point `DATASET_DIR=/path/to/be-mlsys-assignment-dataset` somewhere else if needed.

## Quick checks (seconds to minutes; good for a live walkthrough)

Run these from `experiments/model_selection/` unless noted.

| Command | Time | What it shows |
|---|---|---|
| `python make_tables.py` | ~1 s | Rebuilds every table in the model-selection note from the stored JSONs (output = `tables.md`) |
| `python probe_bench.py --force resnet18_in1k@64` | ~20 s | Re-runs one frozen-backbone probe end-to-end (features → CV → eval → FLOPs → latency) |
| `python specialized.py --force huygens_resnet50` | ~1.5 min | The "undocumented preprocessing" finding: 99.0% with raw 64 px input, 12.9% with standard preprocessing |
| `python finetune.py resnet18.tv_in1k 64 1e-4 16` | ~12 min | One full CPU fine-tune |
| `cd ../data_analysis && python baseline.py` | ~10 s | The 85% colour/texture baseline and its coverage-vs-accuracy numbers |

I verified that re-runs reproduce the stored accuracy numbers **exactly** on CPU. Latency numbers move by roughly ±10–15% between runs.

## Full reproduction, in order

Rough times are on a 64-core AMD EPYC 7542; each step writes JSON next to the script.

1. **Frozen probes + zero-shot:** `python probe_bench.py` → `feats/<key>.json`. Takes 1.5–2 h; the DINOv3-SAT ViT-L and the SSL4EO scale search dominate.
2. **Fine-tunes:** `ft/<model>@<size>.json`, ~20–80 min each at 16–24 threads. The exact runs:
   ```bash
   python finetune.py resnet18.tv_in1k 224 1e-4 24
   python finetune.py resnet18.tv_in1k 64 1e-4 24
   python finetune.py vit_small_patch16_224.augreg_in21k_ft_in1k 224 5e-5 24
   python finetune.py vit_small_patch16_224.augreg_in21k_ft_in1k 112 5e-5 16
   python finetune.py vit_small_patch16_dinov3.lvd1689m 112 5e-5 16
   python finetune.py efficientvit_b0.r224_in1k 224 3e-4 16
   python finetune.py mobilenetv3_large_100.ra_in1k 224 3e-4 16
   python finetune.py ssl4eo_resnet18_rgb_moco 224 1e-4 16
   ```
3. **Leakage matching:** `python leakcheck.py` → `leakcheck.json`, ~15 min. It matches each of our 1,260 tiles to the train/val/test split of the public EuroSAT copies the checkpoints were trained on.
4. **Specialized checkpoints:** `python specialized.py` → `spec/<key>.json`, ~45 min. It needs `leakcheck.json` for the held-out columns.
5. **Latency (idle machine only):**
   ```bash
   python latency.py resnet18_in1k resnet18_in1k@64 resnet50_in1k efficientnet_b0 mobilenetv3_large convnext_tiny_in22k \
     vit_small16_in21k deit_tiny16 swin_tiny_in22k efficientvit_b0 fastvit_t8 tiny_vit_5m mobilevitv2_100 dinov2_vits14 \
     clip_vitb32_openai georsclip_vitb32 satlas_swinv2t_s2_rgb ssl4eo_resnet18_rgb_moco ssl4eo_resnet50_rgb_moco \
     dinov3_vitb16 dinov3_vitl16_sat   # these two: PyTorch timing only; ONNX export fails without latency_fix's switch
   python latency_fix.py          # ViT-S@112 and DINOv3 ViT-S @64/112/224 (see "ONNX export" below)
   python specialized.py --latency
   ```
6. **Tables:** `python make_tables.py`.

## Which result files feed which table

| Table in `model_selection_thoughtprocess.md` | Source |
|---|---|
| §4.1 zero-shot, §4.3 linear probes | `feats/*.json` (+ `latency/*.json` for the ms columns) |
| §4.2 specialized checkpoints and leakage | `spec/*.json`, `leakcheck.json` |
| §4.4 fine-tunes | `ft/*.json` (+ `feats/` for the LP column, `latency/` for ms) |
| §7 finalists | `feats/` + `ft/` + `latency/` |

## Things worth knowing

- **The eval set is never used to choose anything.** Probes pick the regularisation by 5-fold CV on `candidate_tiles`. Fine-tunes pick the epoch on a 15% validation split of `candidate_tiles`. `eval_set` is scored once.
- **Latency needs an idle CPU.** Thread counts are set inside the scripts (`torch.set_num_threads`, ONNX Runtime `intra_op_num_threads`). The fine-tune training times were measured while other jobs shared the machine, so treat them as upper bounds.
- **ONNX export of ViTs.** Build the model at a fixed input size (`img_size=112`), not with `dynamic_img_size=True`, whose runtime position-embedding resampling isn't exportable. DINOv3 additionally needs `timm.layers.set_fused_attn(False)` for the TorchScript exporter. `latency_fix.py` does both; the weights are unchanged.
- **Fine-tuned weights are not saved.** These are benchmark runs; the Part 2 training script is where the production model gets trained, exported and checksummed.
- **Preprocessing variants.** `specialized.py` scores each checkpoint with its own declared pipeline (`native`) and with a generic resize (`plain`), plus a small search when nothing is documented. `by_prep` in each JSON shows the spread.
- **Known failures, kept on purpose:**
  - `nielsr/van-base-…` fails to load (the VAN architecture was removed in transformers 5).
  - `Adilbai/EuroSAT-Swin` runs but stays at chance under every preprocessing tried.
- **Not included:** the `feats/*.npz` feature caches (~49 MB, regenerated by step 1) and exported `.onnx` files (created and deleted per model by `latency.py`).

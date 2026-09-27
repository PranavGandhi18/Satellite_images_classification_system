# Model Selection: Thought Process

*GalaxEye take-home, companion to `DESIGN_NOTE.md` §4.1. This note replaces the "Which model" decision there with a measured one.*

## TL;DR

- **What I tested.** I benchmarked **58 model configurations on this exact data**, all on CPU:
  - **19 specialized satellite classifiers** downloaded as-is (18 loadable);
  - **31 frozen backbone configurations**, including remote-sensing foundation models, DINOv2/v3, CLIP-family models, CNNs and efficient transformers;
  - **8 full fine-tunes**, including one Sentinel-2-pretrained ResNet;
  - with ONNX Runtime latency on an idle machine.
- **Is training necessary?**
  - Zero-shot models reach ~71% at best.
  - Specialized EuroSAT checkpoints look near-perfect (97.7–100% on tiles they never saw). But **81–91% of our eval tiles were in their training data**, their preprocessing is often undocumented (one drops from 99% to 13% with standard preprocessing), and the best ones have no declared licence.
  - Training *our own* model is cheap: **a linear head in seconds** (up to 97.6%), and **a full fine-tune in ~25 minutes on CPU** (up to 98.9%).
  - Owning it is what makes it measurable, calibratable and retrainable offline.
- **ResNet vs transformer.**
  - My earlier ResNet-18 pick was an unmeasured default.
  - Frozen transformer features are better *when their pretraining is bigger*; a Sentinel-2-pretrained ResNet matches them.
  - Once fine-tuned, the top backbones all land at 98–99%, within noise of each other.
  - At 112 px input a ViT-S is **as fast on CPU as ResNet-18 @224** (31 vs 41 ms on 1 thread, ~100 tiles/s both). Attention is only ~2% of its compute at 49 patches. "Too slow on CPU" doesn't apply here.
- **Pick:** a **ViT-S/16 fine-tuned at 112 px, served via ONNX**.
  - Initialised from **DINOv3** if its licence clears legal review.
  - Otherwise from the **Apache-2.0 ImageNet-21k ViT-S/16**: same code path, −0.8 pp nominal.
  - Alternatives: **SSL4EO ResNet-18** is an equally good CNN option; **EfficientViT-B0** is the pick for weak hardware (4 ms per tile, 97.0%).

---

## 1. What the model has to satisfy

Before looking at any leaderboard, I wrote down what "good" means for this service. Accuracy is only one line of the list.

| Requirement | Where it comes from | What it rules in or out |
|---|---|---|
| **Runs on CPU, offline** | Brief: "locally runnable, pre-trained model you can run on CPU", "no internet". | Weights vendored with the release, a few hundred MB at most, tens of milliseconds per tile, clean ONNX export. |
| **Input is 64×64, 8-bit RGB** | The data: Sentinel-2 true colour, no other bands, no metadata. | Most remote-sensing foundation models want 13-band multispectral or SAR input, so they are out for *this* data. RGB-capable models stay in. |
| **7 classes, 1,050 labelled tiles** | `candidate_tiles/` | A small-data regime: pretraining quality matters more than architecture tricks. Nothing big can be trained from scratch. |
| **Honest evaluation** | The brief says reasoning matters more than accuracy. | The model must not have seen the 210 eval tiles in training. That rules out taking public EuroSAT checkpoints' scores on our eval set at face value. |
| **Useful confidence and embeddings** | Design note §4.2 (needs-review and unfamiliar statuses). | We need calibrated probabilities and features in which "distance to training data" means something. |
| **Licence fits a commercial product** | GalaxEye is a company. | Non-commercial weights (CC-BY-NC) are out. Custom licences need a legal check. |
| **Maintainable and retrainable on the box** | Offline, with analyst corrections flowing back. | Mainstream tooling (timm / PyTorch → ONNX), and a training run cheap enough for CPU. |

---

## 2. The landscape: state of the art for this problem (September 2026)

A background research pass read the primary sources (papers, model cards, torchgeo and timm source) and marked which facts were verified. The summary below is condensed from that. **Important:** papers use different EuroSAT splits (10k/5k, 21.6k/5.4k, GEO-Bench's 2k-train "m-eurosat", VTAB's 1k-train), so numbers are comparable only *within* a row group, not across it.

### 2.1 Specialized, already-trained satellite land-use classifiers (use as-is, no training)

This is the "just download a model that already does exactly this" option. I searched the Hugging Face hub systematically (`eurosat`, `resisc45`, `ucmerced`, `bigearthnet`, `land-cover`, `sentinel-2`, `satellite`, `remote-sensing`, `scene-classification`, …: 514 hits). torchgeo ships pretrained *backbones*, not EuroSAT classifier heads. The hits fall into four groups:

| Group | What exists | CPU? | Used here? |
|---|---|---|---|
| **A. Trained on EuroSAT itself** (10 classes, a superset of our 7) | About 60 repos, most of them Swin-T clones of one Hugging Face tutorial. Distinct architectures: ResNet-18/50, ConvNeXt-T/B, ConvNeXt-V2-T, Swin-T/B, ViT-B/16, ViT-L/16, VAN-B, and CLIP ViT-B/32 and B/16 fine-tunes. | yes (ViT-L only just) | **13 distinct models tried**, 12 loadable (§4.2) |
| **B. Trained on other land-use datasets whose classes overlap ours** | RESISC45 (45 aerial scene classes: forest, river, lake, freeway, industrial area, dense/medium/sparse residential, farmland…), GID-15 (Gaofen-2, 4 m), RSSCN7, UC Merced | yes | **6 tested**, each with an explicit label mapping onto our 7 classes |
| **C. Trained on Sentinel-2 but needing multispectral input** | BigEarthNet v2 classifiers (BIFOLD, 19 multi-label classes, 10–12 bands); Dynamic World and Esri 10 m land cover (segmentation, 6–9 bands) | yes | **No**: our PNGs have only RGB, so feeding them would be garbage in |
| **D. Vision-language models fine-tuned for EuroSAT** | GeoQwen-VL-2B-EuroSAT, LFM2.5-450M EuroSAT LoRA, remote-sensing Qwen2-VL-2B | technically yes, at seconds per tile | **No**: 0.45–2B-parameter generative models for a 7-way label is the wrong tool on CPU |

A cautionary finding from the search: **4 repos named "…-finetuned-eurosat" are not EuroSAT models at all.** Their configs contain Food-101 labels, skin-lesion labels, "defense / non-defense images" and "Up / Down". A model's name is not evidence of what it contains, which matters when weights are carried onto an air-gapped machine.

### 2.2 Remote-sensing foundation models

**Usable here** (accept RGB and are realistic on CPU):

| Model | Arch / size | Pretraining | Licence | Published EuroSAT |
|---|---|---|---|---|
| **SSL4EO-S12** RGB weights (torchgeo) | ResNet-18 / ResNet-50 | MoCo-v2 on ~1M Sentinel-2 patches | CC-BY-4.0 | ResNet-50 RGB: linear probe 96.6, fine-tune 98.0 |
| **SeCo** RGB weights (torchgeo) | ResNet-18 / 50 | Seasonal contrast on Sentinel-2 RGB | Apache-2.0 | ResNet-18: 93.1 |
| **SatlasPretrain** Sentinel-2 RGB (torchgeo) | ResNet-50, Swin-v2-T | Supervised multi-task, 302M labels | ODC-BY (sources conflict) | m-eurosat fine-tune 96.3 |
| **DINOv3-SAT** (Meta, Aug 2025) | ViT-L/16, 300M (plus a 7B teacher) | DINOv3 on 493M Maxar RGB tiles at **0.6 m** | DINOv3 License (custom) | m-eurosat linear probe 94.1 (the *web* 7B model scores 97.0) |
| **DOFA**, **Panopticon**, **Copernicus-FM** | ViT-B (~86–98M) | Any-band / any-sensor models | CC-BY-4.0 / Apache-2.0 | m-eurosat LP 93.9–96.4 (multispectral) |
| **TerraMind 1.0** (IBM/ESA, 2025) | tiny → large | Any-to-any generative, S1 / S2 / RGB / DEM | Apache-2.0 | m-eurosat kNN 85.6–90.0 |

**Excluded for this data, but relevant for GalaxEye's multi-sensor future:**

- **CROMA** — S1 + S2; 99.46% EuroSAT fine-tuned, the multispectral state of the art.
- **SSL4EO ViT-S/16** — 13-band only.
- **DeCUR**, **SoftCon** — multispectral / SAR.
- **Prithvi-EO 2.0** — 6-band HLS.
- **Galileo** and **OlmoEarth** — the 0.8–6M-parameter nano and tiny variants are very CPU-friendly, but multispectral.
- **Clay v1.5** — ViT-L.
- **SkySense** — ~2B parameters.
- **Scale-MAE** and **SatMAE** — ViT-L; Scale-MAE is also CC-BY-NC.

### 2.3 Remote-sensing vision-language models (zero-shot, no training)

Published zero-shot accuracy on 10-class EuroSAT. Protocols vary a lot: RemoteCLIP ViT-B/32 alone is reported anywhere from 31 to 45% depending on the paper. The 2026 unified comparison by Baltzi et al. is the fairest single source.

| Model | Published zero-shot EuroSAT |
|---|---|
| OpenAI CLIP ViT-B/32 / L/14 | 49.4 / 59.9 (CLIP paper) |
| RemoteCLIP ViT-B/32 | 36.0 — *below* plain CLIP in its own paper |
| GeoRSCLIP ViT-B/32 / H/14 | 61.5 / 67.5 |
| SkyCLIP ViT-L/14 | 69.7 (Baltzi 2026, best in that study) |
| SenCLIP ViT-B/32 (2025) | 71.2 |
| GRAFT ViT-B/16 | 63.8 (CC-BY-NC) |

**Ceiling: roughly 60–75%.** These models are useful as a sanity check or a labelling aid, not as the classifier.

### 2.4 General-purpose pretrained backbones (used as feature extractors)

| Family | Sizes realistic on CPU | Licence | Notes |
|---|---|---|---|
| **DINOv3** (Meta, Aug 2025) | ViT-S/16 21M, S+ 29M, B 86M; ConvNeXt-T/S distilled | DINOv3 License: commercial use allowed with attribution (needs legal check) | Self-distillation on 1.7B images; state-of-the-art frozen features. |
| **DINOv2** | ViT-S/14 21M, B/14 86M | Apache-2.0 | On RGB m-eurosat, a frozen DINOv2 ViT-B beat several multispectral remote-sensing models (95.5 vs 91–93). |
| **CLIP**, **SigLIP2** | ViT-B/32, B/16 | MIT / Apache-2.0 | CLIP paper linear probe on EuroSAT: L/14 98.2, B/32 97.0. |
| **ImageNet-21k ViT** (timm `augreg`) | ViT-S/16 22M, ViT-B/16 86M | Apache-2.0 | The standard supervised ViT. |
| **ImageNet CNNs** | ResNet-18/50, ConvNeXt-T, EfficientNet-B0, MobileNetV3/V4 | BSD / Apache-2.0 | CLIP paper linear probe on EuroSAT: ImageNet ResNet-50 96.7, EfficientNet-B0 97.3. |

### 2.5 CPU-efficient architectures, including efficient transformers

Published timm CPU benchmark (i7-12700H, fp32, batch 1, at 224 px):

| Model | Type | Params | GMACs | CPU ms |
|---|---|---|---|---|
| MobileNetV3-L | CNN | 5.5M | 0.23 | 7.4 |
| MobileNetV4-Conv-M | CNN | 9.7M | 0.84 | 11.8 |
| EfficientNet-B0 | CNN | 5.3M | 0.40 | 10.7 |
| **EfficientViT-B0** (MIT Han Lab) | conv + linear-attention hybrid | 3.4M | 0.10 | 4.7 |
| FastViT-T8 | hybrid | 4.0M | 0.70 | 15.5 |
| TinyViT-5M | hybrid | 5.4M | 1.28 | 21.0 |
| MobileViTv2-1.0 | hybrid | 4.9M | 1.84 | 20.6 |
| DeiT-Tiny | ViT | 5.7M | 1.26 | 12.9 |
| ResNet-18 | CNN | 11.7M | 1.82 | 16.0 |
| ViT-S/16 | ViT | 22.1M | 4.61 | 37.4 |
| ResNet-50 | CNN | 25.6M | 4.11 | 32.8 |

### 2.6 Where the ceiling is

- **Multispectral, fine-tuned:** 99.46% (CROMA ViT-L).
- **RGB, fine-tuned:** 98.6% (original EuroSAT paper, ResNet-50, 80/20 split); up to ~99.2% in preprints.
- **Low-data regime:** VTAB-1k uses only 1,000 training images, like our 1,050. There, an ImageNet ResNet-50 scores linear probe 91.8 / fine-tune 95.8, and a ViT-L/16 fine-tunes to 95.6.

So anything around 98% on this data is at the published ceiling.

---

## 3. How I tested: one protocol for every model

Reading leaderboards isn't enough, because every paper uses a different split, resolution and preprocessing. So I ran the candidates myself on *this* data, all under the same rules:

- **Training data:** only the 1,050 `candidate_tiles`.
- **Two accuracy numbers per model:**
  - **5-fold cross-validation on the candidates (CV, n = 1,050).** This is the more stable one, about ±1.5 pp.
  - **One score on the 210 eval tiles.** About ±3 pp, because 210 is small.
  - The eval set was never used to choose anything.
- **Linear probe (LP):** freeze the pretrained backbone, take its pooled features, and fit a logistic regression. The regularisation strength is picked by CV. Training takes seconds.
- **Full fine-tune (FT):** retrain all weights with one fixed recipe, chosen up front and never tuned on eval:
  - AdamW, 15 epochs, one-cycle cosine learning rate, label smoothing 0.1;
  - random flips and 90° rotations (overhead imagery has no "up");
  - 15% of the candidates held out as validation, to pick the best epoch.
  - Everything was trained **on CPU only** (16–24 threads).
- **Input size:** the 64 px tiles are upsampled with bicubic interpolation to the model's native 224 px. I also tested 64 and 112 px, because input size drives CPU cost.
- **CPU cost:**
  - parameter count, and GMACs from PyTorch's FLOP counter;
  - latency at batch 1 with 1 and 4 threads, in PyTorch and in **ONNX Runtime** (what the service would actually run);
  - measured on an otherwise idle AMD EPYC 7542 at 2.9 GHz.
- **Zero-shot** (CLIP-style models only): prompt ensembles such as "a satellite photo of {class}", using the CLIP paper's EuroSAT class names.
- **Off-the-shelf EuroSAT checkpoints:** run as-is, with their 10-class output restricted to our 7 classes.

---

## 4. Results, from least to most training

Four levels of effort, from nothing at all to retraining every weight.

### 4.1 Level 0: zero-shot (no training at all)

| Model | Zero-shot, 7-way (no training) | Same encoder + linear probe |
|---|---|---|
| GeoRSCLIP ViT-B/32 | 71.4 | 97.6 |
| SigLIP2 ViT-B/16 | 65.2 | 97.6 |
| CLIP ViT-B/32 | 54.8 | 94.8 |
| RemoteCLIP ViT-B/32 | 45.2 | 92.9 |

A CLIP model fine-tuned on UC Merced and used zero-shot with our class names reaches 70.0% (§4.2). **The ceiling without any training is about 70%**, in line with the published 60–75%. The last column is the point: the *same* frozen encoders reach 93–98% once a 7-way linear head is fitted on our 1,050 tiles. The knowledge is in the features; what's missing is the mapping to our labels.

### 4.2 Level 1: specialized pretrained classifiers, used as-is

Each model's labels were mapped onto our 7 classes; source labels with no counterpart (e.g. Pasture, meadow) count as wrong. Because we trained none of these, **all 1,260 of our tiles are test data for them**.

| HF repo | Arch | Trained on | Covers our classes | Acc. eval (n=210) | Acc. all 1,260 | Acc. with plain resize | Top-1 is a usable label | Held-out estimate† | torch ms, 1 / 4 threads |
|---|---|---|---|---|---|---|---|---|---|
| `tanganke/clip-vit-base-patch16_eurosat` | CLIP ViT-B/16 (fine-tuned) | EuroSAT (tanganke/eurosat 21.6k/2.7k) | 7/7 | 100.0 | 100.0 | 100.0 | 99.9 | 100.0 (n=116) | 518 / 203 |
| `tanganke/clip-vit-base-patch32_eurosat` | CLIP ViT-B/32 (fine-tuned) | EuroSAT (tanganke/eurosat 21.6k/2.7k) | 7/7 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 (n=116) | 169 / 70 |
| `tanganke/convnext-base-224_eurosat_sgd_batch-size-64_lr-0.01_steps-4000` | ConvNeXt-B | EuroSAT (tanganke/eurosat) | 7/7 | 99.5 | 99.9 | 99.6 | 100.0 | 99.1 (n=116) | 528 / 220 |
| `bknyaz/vitl-eurosat-in21k` | ViT-L/16 | EuroSAT (jonathan-roberts1, 90/10) | 7/7 | 100.0 | 99.8 | 99.7 | 99.9 | – | 1860 / 689 |
| `cm93/resnet50-eurosat` | ResNet-50 | EuroSAT (cm93/eurosat 80/10/10) | 7/7 | 99.5 | 99.7 | 94.8 | 99.2 | 97.7 (n=132) | 126 / 58 |
| `nielsr/swin-tiny-patch4-window7-224-finetuned-eurosat` | Swin-T | EuroSAT (nielsr/eurosat-demo, random 90/10) | 7/7 | 99.0 | 99.2 | 99.1 | 99.4 | – | 164 / 81 |
| `huygens-jnr/eurosat-resnet50` | ResNet-50 | EuroSAT (split unknown) | 7/7 | 99.0 | 99.0 | 12.9 | 99.2 | – | 28 / 26 |
| `cm93/resnet18-eurosat` | ResNet-18 | EuroSAT (cm93/eurosat 80/10/10) | 7/7 | 97.1 | 98.7 | 96.4 | 99.0 | 97.7 (n=132) | 54 / 21 |
| `nielsr/vit-finetuned-eurosat-kornia` | ViT-B/16 | EuroSAT (split unknown) | 7/7 | 99.0 | 98.7 | 99.0 | 99.1 | – | 528 / 211 |
| `mrm8488/convnext-tiny-finetuned-eurosat` | ConvNeXt-T | EuroSAT (nielsr/eurosat-demo, random 90/10) | 7/7 | 98.6 | 98.7 | 98.7 | 98.9 | – | 154 / 69 |
| `taufiqdp/convnext-eurosat` | ConvNeXt-V2-T @384 | EuroSAT (split unknown) | 7/7 | 91.0 | 90.2 | 88.5 | 97.4 | – | 715 / 318 |
| `tanganke/clip-vit-base-patch32_resisc45` | CLIP ViT-B/32 (fine-tuned) | RESISC45 (aerial, 0.2-30 m) | 7/7 | 74.3 | 73.2 | 73.4 | 34.9 | – | 172 / 70 |
| `NemesisAlm/clip-fine-tuned-satellite` | CLIP ViT-B/32 (fine-tuned), zero-shot head | UC Merced (aerial, 0.3 m); zero-shot with our class names | 7/7 | 70.0 | 69.3 | 68.1 | 100.0 | – | 176 / 74 |
| `prithivMLmods/RESISC45-SigLIP2` | SigLIP2-B | RESISC45 (aerial, 0.2-30 m) | 7/7 | 52.4 | 51.3 | 58.3 | 10.2 | – | 553 / 227 |
| `SeyedAli/Remote-Sensing-UAV-image-classification` | ViT-B/16 | RSSCN7 (aerial) | 6/7 (no Highway) | 48.1 | 43.4 | 43.4 | 93.9 | – | 512 / 202 |
| `tanganke/convnext-base-224_resisc45_sgd_batch-size-64_lr-0.01_steps-4000` | ConvNeXt-B | RESISC45 (aerial, 0.2-30 m) | 7/7 | 40.5 | 37.5 | 42.8 | 17.8 | – | 509 / 208 |
| `prithivMLmods/GiD-Land-Cover-Classification` | SigLIP2-B | GID-15 (Gaofen-2, 4 m) | 7/7 | 29.5 | 28.7 | 29.8 | 87.1 | – | 557 / 227 |
| `Adilbai/EuroSAT-Swin` | Swin | EuroSAT (nielsr/eurosat-demo) | 7/7 | 18.1 | 14.3 | 14.3 | 61.3 | – | 163 / 80 |

**†** Accuracy on the subset of our tiles that sit in that checkpoint's own *test* split, found by exact pixel-hash matching against the public dataset copy it was trained on.

Two special cases:
- `nielsr/van-base-finetuned-eurosat-imgaug` is missing from the table because it no longer loads: its VAN architecture was removed in transformers 5.
- `Adilbai/EuroSAT-Swin` *is* in the table, at chance level (11–14% under every preprocessing I tried). Its card is internally inconsistent: it carries SAR tags but was trained on RGB EuroSAT.

**Did these models train on our eval tiles?** I matched our tiles pixel-for-pixel against the public EuroSAT copies the checkpoints name:

| Public copy they trained on | Checkpoints | Our 210 eval tiles in its **train** split | In its **test** split |
|---|---|---|---|
| `cm93/eurosat` | cm93 ResNet-18 / 50 | **171 (81%)** (+16 in validation) | 23 |
| `tanganke/eurosat` | tanganke CLIP-B/32, B/16, ConvNeXt-B | **191 (91%)** | 19 |
| `nielsr/eurosat-demo` | mrm8488 ConvNeXt-T, nielsr Swin-T / ViT-B, Adilbai Swin | all 210 (the whole 27k pool; the tutorial splits it randomly 90/10) | — |

What this shows:

1. **The best specialized checkpoints really are excellent.** tanganke's EuroSAT fine-tuned CLIP ViT-B/32 and B/16 get **116/116 right on tiles from their own held-out split** (95% CI lower bound ≈ 96.8%). cm93's ResNets get 97.7% on theirs.
2. **Headline numbers are inflated by leakage.** cm93 ResNet-50 scores 99.9% on the 1,022 of our tiles it trained on, but 97.7% on the 132 it didn't.
3. **Preprocessing is a silent failure mode.**
   - cm93 ResNet-50: **99.7%** with its own timm pipeline, 94.8% with a plain resize.
   - `huygens-jnr/eurosat-resnet50`: **99.0%** only with raw 64 px pixels in [0, 1], which is documented nowhere. With standard ImageNet preprocessing it scores **12.9%, below chance**.
   - Wrong preprocessing gives no error, just confident nonsense. This is exactly why the design note ships preprocessing *inside* the model artifact.
4. **"Satellite classifier" ≠ "classifier for this sensor".**
   - Models trained on other scene datasets (RESISC45, UC Merced, GID-15, RSSCN7: 0.2–30 m per pixel, 256–600 px scenes) don't transfer to 64 px Sentinel-2 tiles at 10 m. Restricted to our classes, they reach **only 29–73%**.
   - The three RESISC45 models put their top-1 on a label we can use only 10–35% of the time (meadow, desert, wetland, …).
   - The GID-15 SigLIP2 maps onto all 7 of our classes, yet still scores just 29%.
5. **Quality varies wildly within "EuroSAT models":** from 100% (tanganke CLIP) to 90% (ConvNeXt-V2-T), to broken (Adilbai), to not EuroSAT at all (4 mislabelled repos, §2.1).

### 4.3 Level 2: frozen pretrained backbone + a trained linear head (seconds of training)

Sorted by cross-validated accuracy (CV). The latency columns come from the idle-machine ONNX Runtime pass. A "–" means that configuration wasn't timed: the large ViTs, DINOv3-B/L (which need the same export switch as §6.2), and a few non-finalists.

| # | Backbone | Type | Pretraining | Input | Params (M) | GMACs | LP CV acc (n=1,050) | LP eval acc (n=210) | ONNX ms, 1 / 4 threads | Licence |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GeoRSCLIP ViT-B/32 | Transformer | CLIP + RS5M captions | 224 | 87.8 | 2.95 | 98.3 | 97.6 | 111.3 / 39.4 | repo MIT, weights "cc" |
| 2 | SSL4EO-S12 ResNet-18 (RGB) | CNN | MoCo on 1M Sentinel-2 patches | 224 | 11.2 | 1.81 | 97.1 | 93.8 | 40.2 / 10.2 | CC-BY-4.0 |
| 3 | ViT-S/16 | Transformer | ImageNet-21k supervised | 224 | 21.7 | 4.24 | 97.1 | 96.7 | 104.9 / 48.2 | Apache-2.0 |
| 4 | SSL4EO-S12 ResNet-50 (RGB) | CNN | MoCo on 1M Sentinel-2 patches | 224 | 23.5 | 4.09 | 96.9 | 96.7 | 87.5 / 25.4 | CC-BY-4.0 |
| 5 | DINOv3 ViT-B/16 | Transformer | Self-supervised, LVD-1689M | 224 | 85.6 | 17.19 | 96.9 | 96.2 | – / – | DINOv3 License |
| 6 | SigLIP2 ViT-B/16 | Transformer | Image-text, WebLI | 224 | 92.9 | – | 96.6 | 97.6 | – / – | Apache-2.0 |
| 7 | DINOv3 ViT-S/16 | Transformer | Self-supervised, LVD-1689M | 224 | 21.6 | 4.33 | 96.5 | 93.8 | 110.2 / 50.0 | DINOv3 License |
| 8 | Swin-T | Transformer | ImageNet-22k supervised | 224 | 27.5 | 4.49 | 96.4 | 95.7 | 117.1 / 53.4 | MIT |
| 9 | DINOv3-SAT ViT-L/16 | Transformer | Self-supervised, 493M Maxar tiles (0.6 m) | 224 | 303.1 | 60.85 | 96.3 | 96.7 | – / – | DINOv3 License |
| 10 | DINOv2 ViT-B/14 | Transformer | Self-supervised, LVD-142M | 224 | 85.7 | 21.94 | 96.2 | 94.8 | – / – | Apache-2.0 |
| 11 | ConvNeXt-T | CNN | ImageNet-22k supervised | 224 | 27.8 | 4.45 | 95.9 | 92.4 | 122.9 / 44.0 | Apache-2.0 |
| 12 | DINOv3 ViT-S/16 @112 px | Transformer | Self-supervised, LVD-1689M | 112 | 21.6 | 1.16 | 95.8 | 96.7 | 30.7 / 14.4 | DINOv3 License |
| 13 | RemoteCLIP ViT-B/32 | Transformer | CLIP + remote-sensing captions | 224 | 87.8 | 2.95 | 95.1 | 92.9 | – / – | Apache-2.0 |
| 14 | TinyViT-5M | Hybrid | ImageNet-22k distilled | 224 | 5.1 | 1.25 | 95.0 | 92.4 | 40.5 / 20.3 | Apache-2.0 |
| 15 | CLIP ViT-B/32 | Transformer | Image-text, 400M pairs | 224 | 87.8 | 2.95 | 94.8 | 94.8 | 110.9 / 39.7 | MIT |
| 16 | DINOv2 ViT-S/14 | Transformer | Self-supervised, LVD-142M | 224 | 21.6 | 5.51 | 94.8 | 95.7 | 138.5 / 57.0 | Apache-2.0 |
| 17 | EfficientViT-B0 | Hybrid | ImageNet-1k supervised | 224 | 2.1 | 0.10 | 94.8 | 94.3 | 4.0 / 3.9 | Apache-2.0 |
| 18 | DINOv3 ViT-S/16 @64 px | Transformer | Self-supervised, LVD-1689M | 64 | 21.6 | 0.45 | 94.5 | 94.3 | 15.2 / 7.5 | DINOv3 License |
| 19 | EfficientNet-B0 | CNN | ImageNet-1k supervised | 224 | 4.0 | 0.38 | 94.3 | 93.8 | 16.4 / 7.6 | Apache-2.0 |
| 20 | MobileNetV3-L | CNN | ImageNet-1k supervised | 224 | 4.2 | 0.22 | 94.1 | 94.8 | 7.3 / 4.2 | Apache-2.0 |
| 21 | ResNet-18 | CNN | ImageNet-1k supervised | 224 | 11.2 | 1.81 | 93.8 | 92.4 | 41.0 / 10.9 | BSD-3 |
| 22 | DINOv2 ViT-S/14 @112 px | Transformer | Self-supervised, LVD-142M | 112 | 22.1 | 1.39 | 93.6 | 92.9 | – / – | Apache-2.0 |
| 23 | FastViT-T8 | Hybrid | ImageNet-1k supervised | 224 | 3.3 | 0.53 | 93.6 | 90.5 | 22.6 / 10.4 | Apple (custom) |
| 24 | ViT-S/16 @112 px | Transformer | ImageNet-21k supervised | 112 | 21.7 | 1.08 | 93.3 | 95.7 | 28.6 / 13.3 | Apache-2.0 |
| 25 | ResNet-50 | CNN | ImageNet-1k supervised | 224 | 23.5 | 4.09 | 93.1 | 96.2 | 92.4 / 28.9 | Apache-2.0 |
| 26 | SatlasPretrain Swin-v2-T (S2 RGB) | Transformer | Supervised multi-task, Sentinel-2 | 256 | 27.6 | 5.94 | 93.0 | 94.3 | 163.1 / 83.5 | ODC-BY |
| 27 | DINOv3 ConvNeXt-T | CNN | Distilled from DINOv3 7B | 224 | 27.8 | 4.45 | 93.0 | 94.3 | – / – | DINOv3 License |
| 28 | MobileViTv2-1.0 @256 px | Hybrid | ImageNet-1k supervised | 256 | 4.4 | 1.81 | 92.9 | 94.8 | 49.3 / 22.9 | Apple (custom) |
| 29 | DeiT-Tiny | Transformer | ImageNet-1k supervised | 224 | 5.5 | 1.07 | 91.8 | 91.9 | 31.3 / 18.6 | Apache-2.0 |
| 30 | SatlasPretrain ResNet-50 (S2 RGB) | CNN | Supervised multi-task, Sentinel-2 | 224 | 23.5 | 4.09 | 90.9 | 89.0 | – / – | ODC-BY |
| 31 | ResNet-18 @64 px | CNN | ImageNet-1k supervised | 64 | 11.2 | 0.15 | 90.5 | 88.6 | 7.1 / 1.3 | BSD-3 |

What stands out:

- **Best frozen features:**
  - GeoRSCLIP ViT-B/32 (CLIP further trained on 5M remote-sensing image-text pairs): **98.3% CV**;
  - then SSL4EO ResNet-18 (Sentinel-2 self-supervised) and ViT-S/16 (ImageNet-21k): 97.1;
  - then SSL4EO ResNet-50 and DINOv3 ViT-B: 96.9.
- **Weakest:** small ImageNet-1k models (ResNet-18 93.8, DeiT-Tiny 91.8) and Satlas ResNet-50 (90.9).
- **"Satellite-pretrained" helps only when the imagery matches.**
  - SSL4EO, pretrained on the *same sensor at the same 10 m resolution*, lifts ResNet-18 from 93.8 to 97.1.
  - DINOv3-SAT, pretrained on 0.6 m Maxar imagery, is no better than the web-trained ViT-S (96.3 vs 96.5) at 14× the compute.
  - Satlas (90.9–93.0) is *worse* than general-purpose models.
  - Among remote-sensing CLIPs, RemoteCLIP barely beats plain CLIP (95.1 vs 94.8), while GeoRSCLIP is clearly better (98.3).
- **Input resolution matters, and not monotonically for ViTs.**
  - ResNet-18: 90.5 CV at 64 px vs 93.8 at 224.
  - DINOv3 ViT-S/16: 94.5 at 64 px (16 patches), 95.8 at 112 (49), 96.5 at 224 (196). On eval, 112 px scores *best* (96.7).
  - 112 px is the sweet spot for cost.
- **Scores are close together.** CV has an uncertainty of about ±1–1.5 pp at n = 1,050, and eval about ±2.5–3 pp at n = 210. Rows within ~1.5 points of each other should be treated as ties. Where CV and eval disagree (e.g. ResNet-50: 93.1 CV vs 96.2 eval), trust CV.

### 4.4 Level 3: full fine-tuning, CPU only

Same fixed recipe for every model (§3). Ranked by correct predictions on validation + eval pooled (368 tiles), since each set alone is too small.

| Fine-tuned model | Type | Input | LP eval → **FT eval** (n=210) | FT val (n=158) | Val+eval correct (n=368) | 95% CI | Acc. on 80% most-confident | CPU train time (threads) | ONNX ms, 1 / 4 threads |
|---|---|---|---|---|---|---|---|---|---|
| DINOv3 ViT-S/16 | Transformer | 112 | 96.7 → **98.6** | 99.4 | 364/368 = **98.9** | 97.2–99.6 | 100.0 | 23 min (16) | 30.7 / 14.4 |
| ViT-S/16 (IN-21k) | Transformer | 224 | 96.7 → **98.1** | 98.7 | 362/368 = **98.4** | 96.5–99.3 | 100.0 | 79 min (24) | 104.9 / 48.2 |
| ResNet-18 | CNN | 224 | 92.4 → **98.6** | 97.5 | 361/368 = **98.1** | 96.1–99.1 | 100.0 | 22 min (24) | 41.0 / 10.9 |
| SSL4EO ResNet-18 (Sentinel-2) | CNN | 224 | 93.8 → **97.6** | 98.7 | 361/368 = **98.1** | 96.1–99.1 | 99.4 | 22 min (16) | 40.2 / 10.2 |
| ViT-S/16 (IN-21k) | Transformer | 112 | 95.7 → **97.6** | 98.7 | 361/368 = **98.1** | 96.1–99.1 | 100.0 | 26 min (16) | 28.6 / 13.3 |
| EfficientViT-B0 | Hybrid | 224 | 94.3 → **97.1** | 96.8 | 357/368 = **97.0** | 94.7–98.3 | 100.0 | 19 min (16) | 4.0 / 3.9 |
| ResNet-18 | CNN | 64 | 88.6 → **95.2** | 94.9 | 350/368 = **95.1** | 92.4–96.9 | 99.4 | 12 min (24) | 7.1 / 1.3 |
| MobileNetV3-L | CNN | 224 | 94.8 → **94.8** | 94.9 | 349/368 = **94.8** | 92.1–96.7 | 99.4 | 25 min (16) | 7.3 / 4.2 |

What stands out:

- **Every decent backbone converges to about 97–99%, and the top five are statistically tied.** Their confidence intervals overlap almost completely. The spread between #1 (DINOv3 ViT-S @112) and #5 is **3 tiles out of 368**.
- **Fine-tuning erases most of the frozen-feature gap.** ImageNet ResNet-18 goes from the bottom third of §4.3 to tied with the best transformers (LP 92.4 → FT 98.6 on eval).
- **Confidence becomes very trustworthy.** On the 80% most-confident predictions, the fine-tuned models are 99.4–100% accurate. That's what makes the design note's "confident vs needs-review" split work.
- **Resolution still matters for CNNs.** ResNet-18 fine-tuned at native 64 px reaches 95.1% vs 98.1% at 224.
- **One recipe for all has a cost.** MobileNetV3 gained nothing from fine-tuning (94.8 → 94.8), which almost certainly means the shared recipe suits it poorly, not that the architecture can't do better.
- **About the training times:** they were measured while other jobs shared the machine, so read them as upper bounds. ViT-S @224's 79 minutes reflects both that contention and its 4× higher compute per tile than at 112 px.

---

## 5. So is training necessary?

**Short answer:**
- A small amount of training, fitting a head, is necessary.
- A full fine-tune is cheap enough on CPU to be worth doing.
- Downloading a specialized checkpoint *can* give excellent accuracy, but I would not ship one as the production model.

The reasoning, step by step:

1. **A pretrained backbone doesn't output our labels.** DINOv3, CLIP and ResNet produce *features*. Something has to map features to {AnnualCrop, …, SeaLake}. Even the lightest option, a 7-way logistic regression on frozen features (a few thousand numbers, fitted in seconds), is training.
2. **Zero-shot is too weak.** Describing the classes in text and letting a vision-language model match them tops out at about 71% here, and at 60–75% in the literature. Training a linear head on the *same* encoder lifts it to 93–98% (§4.1). So the knowledge is in the features; what's missing is the mapping to our labels.
3. **Specialized EuroSAT checkpoints are accurate, but can't be trusted as-is, for five reasons:**
   - **Can't be measured honestly.** 81–91% of our eval tiles were in their training data, so our eval set cannot measure them. Only the 116–132 of our tiles that sit in their *held-out* splits give an honest number, and even those were probably used by the authors for model selection.
   - **Undocumented preprocessing is a silent failure mode.** One model goes from 99% to 13% with ordinary ImageNet preprocessing. Another loses 5 points with a plain resize. A model you didn't train comes with a preprocessing contract you have to reverse-engineer.
   - **Provenance and licence.** The most accurate ones (the tanganke CLIP fine-tunes) declare **no licence**. Four "EuroSAT" repos contained other models entirely. For a commercial, offline product that's a blocker until someone clears it.
   - **The class set is EuroSAT's, not GalaxEye's.** They carry 10 classes, and GalaxEye's taxonomy will change. Analyst corrections from the review queue can only improve a model we can retrain.
   - **The real inputs will drift away from EuroSAT.** Other processing levels, other sensors, and SAR for GalaxEye will all shift the data. What transfers to that future is a *training pipeline*, not a EuroSAT checkpoint.
4. **Fine-tuning buys accuracy, and a lot of it for some backbones.** Full fine-tuning adds up to +6.6 points over the linear probe (+1.4 at the low end, and nothing for MobileNetV3; see §4.4), and in our runs it made the most confident predictions essentially error-free (§4.4). The gain is largest for ImageNet-1k CNNs (ResNet-18: 92.4 → 98.6) and smallest for backbones whose frozen features are already strong (ViT-S/16: 96.7 → 98.1). This matches the literature: contrastive and self-distilled features lose only 1–3 points under a linear probe; MAE-style features lose more.
5. **Training is cheap enough to live on the box.**
   - A linear probe fits in seconds.
   - A full fine-tune of a 20M-parameter model on 1,050 tiles takes about 20–25 minutes on 16 CPU threads, with no GPU.
   - So retraining after analysts correct a few hundred labels is an overnight job on the same isolated machine. That's a system property, not just a modelling one.

**How the specialized checkpoints *are* useful:**
- as an independent **second opinion** (disagreement between them and our model is a great "needs review" signal, and a drift alarm);
- as a **sanity baseline** in the evaluation report;
- as a **pseudo-labeller** when bootstrapping new data.

In each role the model is a tool, not the thing being served.

---

## 6. Why a transformer and not ResNet (and why I first said ResNet)

### 6.1 Why the design note said ResNet-18

Honestly, it was a **default, not a measurement**. ResNet-18 is:
- the safe reflex: 11M parameters, about 1.8 GMACs, trivial ONNX export, understood by everyone;
- a CNN, with the convolutional inductive bias (locality, translation equivariance) that the original ViT paper showed is helpful when data is small;
- a sidestep of a real worry: a ViT with 16-px patches sees a 64-px tile as only 4×4 = 16 tokens, which sounded too coarse.

Those were reasonable priors, so I tested each of them.

### 6.2 What the measurements say

**Finding 1: as frozen feature extractors, transformers are better, *if* their pretraining is bigger.** Holding the pretraining roughly fixed (linear-probe CV accuracy, n = 1,050):

| Pretraining | CNN | Transformer / hybrid | Verdict |
|---|---|---|---|
| ImageNet-1k, small models | ResNet-18 93.8 · EfficientNet-B0 94.3 · MobileNetV3-L 94.1 | DeiT-Tiny 91.8 · EfficientViT-B0 94.8 · FastViT-T8 93.6 · MobileViTv2 92.9 | no clear winner |
| ImageNet-21k/22k | ConvNeXt-T 95.9 | Swin-T 96.4 · **ViT-S/16 97.1** | transformer +0.5 to +1.2 |
| Same DINOv3 teacher | ConvNeXt-T (distilled) 93.0 | **ViT-S/16 96.5** | transformer +3.5 |
| Sentinel-2 in-domain | **SSL4EO ResNet-18 97.1** · ResNet-50 96.9 | Satlas Swin-v2-T 93.0 (different, supervised recipe) | the in-domain CNN wins |

So the ranking of what matters is:
1. **pretraining data**: scale, and whether it matches our sensor;
2. **then fine-tuning** (Finding 2);
3. **then input resolution;**
4. **only then** CNN vs attention.

Transformers win this benchmark mostly because the best large-scale pretrained checkpoints available today are transformers (ImageNet-21k ViTs, DINOv3, CLIP / SigLIP2 / GeoRSCLIP).

**Finding 2: after fine-tuning, architecture stops mattering much.**
- Pooled val + eval: DINOv3 ViT-S 98.9, ViT-S/16 @224 98.4, ResNet-18 98.1, ViT-S/16 @112 98.1, SSL4EO ResNet-18 98.1.
- All inside each other's confidence intervals.
- My original ResNet-18 choice would have been *fine* on accuracy once fine-tuned. It just isn't the best choice on the other criteria.

**Finding 3: "transformers are data-hungry" is about training from scratch, not fine-tuning.**
- The original ViT paper's point was that without big pretraining, CNNs' built-in inductive bias wins. With pretraining, 150 images per class was plenty: ViT-S/16 fine-tuned to 98.4% on 892 training tiles.
- The literature agrees: on VTAB-1k (1,000 training images), a fine-tuned ViT-L/16 scores 95.6 vs 95.8 for a ResNet-50.

**Finding 4: the "only 16 tokens" worry is real, and the fix is cheap.**
- At native 64 px, a /16 ViT sees a 4×4 grid, and DINOv3 ViT-S loses about 2 points (94.5 CV).
- Upsampling to **112 px gives 7×7 = 49 patches**. Each patch then covers about 9×9 original pixels (≈ 90 m), which recovers the accuracy (96.7 eval, and 98.9 after fine-tuning).
- Going on to 224 px (196 patches) costs 4× more compute for no gain.

**Finding 5: on CPU, attention is not the bottleneck at these sizes.**
- Per layer, self-attention costs about 2·N²·d multiply-adds, while the linear/MLP layers cost about 12·N·d². With d = 384, attention is **≈ 2% of ViT-S compute at 112 px** (N = 50 tokens) and ≈ 8% at 224 px.
- Nearly all the work is dense matrix multiplications (GEMMs), which CPU math libraries (oneDNN, MLAS) run close to peak.
- So FLOPs and latency track well for ViTs, whereas depthwise-conv CNNs like MobileNet have low FLOPs but poor arithmetic intensity.
- The result: **ViT-S/16 at 112 px needs ≈ 1.1 GMACs, less than ResNet-18 at 224 px (1.8 GMACs)**.
- Measured with ONNX Runtime on an idle EPYC core:

  | Model | 1 thread | 4 threads | Batch throughput |
  |---|---|---|---|
  | ViT-S/16 @112 | **28.6 ms** | 13.3 ms | 109 tiles/s |
  | DINOv3 ViT-S @112 | 30.7 ms | 14.4 ms | 99 tiles/s |
  | ResNet-18 @224 | 41.0 ms | 10.9 ms | 104 tiles/s |

  The transformer is faster on one thread, a little slower on four (convolutions spread across threads slightly better), and equal in throughput. **"Transformers are too slow for CPU" does not hold at this input size.**
- It *does* hold at 224 px: ViT-S/16 @224 takes 105 ms (31 tiles/s), 3.5× slower for +0.3 pp. That's why the resolution choice matters more than the architecture choice.
- Both ViTs export to ONNX cleanly (outputs match PyTorch to ≤ 2e-5), with one practical wrinkle:
  - build the model at a fixed input size, rather than timm's `dynamic_img_size`, which resamples position embeddings at runtime;
  - DINOv3 needs timm's non-fused attention path for the TorchScript exporter.

  Both are export-time switches; neither changes the weights.

### 6.3 Where a CNN would still be the right call

- **Very weak hardware** (a small ARM board, or a strict per-tile budget). Tiny mobile models win there: MobileNetV3, a depthwise-conv CNN with mature static INT8 quantisation, or EfficientViT-B0, a conv + linear-attention hybrid and my pick for that case (4 ms per tile, 97.0%).
- **Variable input sizes.** CNNs take any size natively. A ViT exported to ONNX is fixed to the size it was exported at, which is fine here because tiles are always 64 px.
- **Multispectral or SAR inputs later.** Then the choice becomes "which remote-sensing foundation model". A few strong ones are CNNs (the SSL4EO ResNets, DeCUR), though most newer ones are ViTs (CROMA, DOFA, TerraMind; §7). The service's model interface (`image → probs, embedding` + manifest) doesn't care.

---

## 7. Decision

The finalists, all fine-tuned the same way and timed on the same idle machine:

| Candidate (fine-tuned) | Val+eval acc (n=368) | Frozen-feature CV | Params (M) | GMACs | ONNX size (MB) | ONNX ms, 1 thread | ONNX ms, 4 threads | Tiles/s (4 threads, batch 32) | PyTorch ms, 1 thread | Licence |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT-S/16 (IN-21k) @112 | 98.1 | 93.3 | 21.7 | 1.08 | 87 | 28.6 | 13.3 | 109 | 44.0 | Apache-2.0 |
| DINOv3 ViT-S/16 @112 | 98.9 | 95.8 | 21.6 | 1.16 | 87 | 30.7 | 14.4 | 99 | 49.1 | DINOv3 License |
| ViT-S/16 (IN-21k) @224 | 98.4 | 97.1 | 21.7 | 4.24 | 87 | 104.9 | 48.2 | 31 | 139.3 | Apache-2.0 |
| ResNet-18 @224 | 98.1 | 93.8 | 11.2 | 1.81 | 45 | 41.0 | 10.9 | 104 | 53.0 | BSD-3 |
| SSL4EO ResNet-18 @224 | 98.1 | 97.1 | 11.2 | 1.81 | 45 | 40.2 | 10.2 | 105 | 53.9 | CC-BY-4.0 |
| EfficientViT-B0 @224 | 97.0 | 94.8 | 2.1 | 0.10 | 9 | 4.0 | 3.9 | 352 | 11.0 | Apache-2.0 |
| ResNet-18 @64 | 95.1 | 90.5 | 11.2 | 0.15 | 45 | 7.1 | 1.3 | 899 | 17.7 | BSD-3 |

**What that costs at scene scale.** One Sentinel-2 granule (10,980 × 10,980 px) is 29,241 tiles of 64 px. With 4 threads:

| Model | Minutes per granule |
|---|---|
| EfficientViT-B0 | **1.4** |
| ViT-S/16 @112 | **4.5** |
| SSL4EO / ImageNet ResNet-18 @224 | 4.6–4.7 |
| DINOv3 ViT-S @112 | **4.9** |
| ViT-S/16 @224, or the best off-the-shelf model (tanganke CLIP-B/32) | ~16 |
| DINOv3-SAT ViT-L | ~6.4 hours |

Latency per tile is not a problem for any finalist. Throughput per scene is where the choices diverge.

### The pick

> **Architecture: a ViT-S/16 transformer (22M parameters), fine-tuned on our tiles at 112 × 112 input, served as an 87 MB fp32 ONNX model.**
>
> **Weights to start from: DINOv3 ViT-S/16**, provided a legal read of the DINOv3 License clears it. If it doesn't, the drop-in fallback is the **Apache-2.0 ImageNet-21k ViT-S/16**: same architecture, same code, same ONNX graph shape, and a one-line config change. It scores 98.1% vs 98.9%, a difference of 3 tiles in 368 and within noise.

Why this pick, criterion by criterion:

1. **Accuracy: best measured, though tied within noise.**
   - 98.9% on validation + eval pooled (364/368). The most-confident 80% of predictions are 100% correct.
   - Best validation score of any model (99.4%). It joint-tops eval with ResNet-18 (98.6%).
2. **Cost: equal to the cheapest serious option.**
   - 1.16 GMACs; 31 ms per tile on 1 thread, 14 ms on 4; ~100 tiles/s.
   - That's the same as ResNet-18 @224, **faster on a single thread**, and 3.5× cheaper than running the same ViT at 224 px.
3. **Better frozen features at our cost point.**
   - At 112 px, DINOv3's frozen features beat the ImageNet-21k ViT by 2.5 points CV (95.8 vs 93.3). DINOv3 uses rotary position embeddings and a mixed-resolution training phase, which plausibly explains why it degrades less when the input isn't 224 px.
   - That matters twice in the design: the "unfamiliar input" score is built on embeddings, and the retrain-in-seconds linear-probe fallback uses them.
4. **It's the family GalaxEye's future models live in.**
   - Almost every current multi-sensor Earth-observation foundation model is a ViT, especially the optical + SAR ones: CROMA, DOFA, Panopticon, Copernicus-FM, TerraMind, Galileo, OlmoEarth, Prithvi.
   - Standardising the serving path on ViT-shaped ONNX graphs now makes the later move to multispectral or SAR inputs a weights swap, not a new pipeline.
5. **Engineering risk is small and known.** It's a mainstream timm model, and both checkpoints export to ONNX with outputs matching PyTorch. The two export switches are documented in §6.2.

### Alternatives I'd accept, and when

| If… | Then use | Cost of switching |
|---|---|---|
| Legal won't clear the DINOv3 License | **ViT-S/16 (ImageNet-21k, Apache-2.0) @112** | None: same code path; about −0.8 pp nominal |
| A CNN is wanted (simplest stack, smallest file, Sentinel-2-native features) | **SSL4EO-S12 ResNet-18 @224** (CC-BY-4.0). Statistically tied (98.1%), best frozen features of the finalists (97.1 CV), 45 MB, and siblings for 13-band and SAR input exist. | Model file + preprocessing (it expects reflectance-like scaling; 255 ↔ 0.3 was picked by CV) |
| The target box is weak, or scene throughput is the bottleneck | **EfficientViT-B0 @224** (Apache-2.0): 4 ms per tile, 352 tiles/s, 9 MB, 97.0% | About −1.9 pp nominal; retrain once |

The off-the-shelf **tanganke CLIP ViT-B/32 EuroSAT** checkpoint is the best "download and go" model (100% on 116 tiles held out from its own training). I'd keep it **beside** production, not in it: as a second opinion whose disagreement flags tiles for review and signals drift. That needs a licence answer first (none declared), and it's 3.5× slower and 4× bigger than the pick.

**Rejected:**
- **ViT-L-class models** (DINOv3-SAT, bknyaz ViT-L): no gain, 6 hours per scene.
- **Cross-dataset "satellite" classifiers** (RESISC45 / GID / RSSCN7 / UC Merced): 29–73%.
- **Zero-shot models as the classifier:** 45–71%.
- **Any checkpoint without documented preprocessing.**

### What would change this decision

- **Target hardware** (question 6 in the design note). On a small ARM box or with a hard throughput target, I'd switch to EfficientViT-B0.
- **Input type.** Once tiles arrive as 13-band or SAR, the choice moves to a remote-sensing foundation model: CROMA for SAR + optical, DOFA / Copernicus-FM / TerraMind for any-band. Same interface, new weights.
- **More labelled data.** A few thousand labelled tiles from GalaxEye's own imagery would let us separate the tied finalists properly. The current gaps are 1–3 tiles.
- **OOD behaviour.** Next experiment: hold one class out as "unknown" and compare how well each finalist's embedding distance detects it. That's the part of the design note's uncertainty story these accuracy numbers don't cover yet.

---

## 8. Caveats: what these numbers do and don't show

- **Small test sets.** 210 eval tiles means ±2–3 pp at these accuracies, and 158 validation tiles means ±2–3 pp. Differences under ~2 points between the top models are **not** significant; I pool val + eval (368 tiles) for the fine-tuned ranking, and CV (1,050) for the probes.
- **One recipe for every model.** The fine-tuning recipe was fixed up front, not tuned per model. MobileNetV3 in particular probably wanted a different learning rate (it gained nothing from fine-tuning), so its row understates it.
- **Latency hardware.** Measured on one server core family (AMD EPYC 7542, AVX2), fp32, ONNX Runtime 1.30 / PyTorch 2.14. The target hardware is unknown (question 6 in the design note), and the ranking can shift on ARM.
- **Leakage estimates** rely on exact pixel matches against the public dataset copies each checkpoint names. Where a model card doesn't name its data, leakage is likely but unmeasured.
- **Licences** are as stated on model cards and repos, as of September 2026. DINOv3's licence terms are summarised from a secondary source and need a real legal read before shipping.
- **Balanced eval.** Accuracy here is on a balanced 7-class set. Deployed accuracy depends on the real class mix and on how much of the input is none of the 7 classes (design note §4.2).

## Appendix: how to reproduce

All code and raw results are in `experiments/model_selection/`; `experiments/README.md` has setup, run order and timings.
- `probe_bench.py` — frozen-backbone linear probes, zero-shot, FLOPs → `feats/`;
- `finetune.py` — CPU fine-tuning → `ft/`;
- `specialized.py` — off-the-shelf classifiers, label mapping, preprocessing variants → `spec/`;
- `leakcheck.py` — pixel-hash matching against public EuroSAT copies → `leakcheck.json`;
- `latency.py` / `latency_fix.py` — PyTorch vs ONNX Runtime timing (the second rebuilds ViTs with a fixed input size and exportable attention) → `latency/`;
- `make_tables.py` — rebuilds every table in this note from those JSONs (`python make_tables.py`, ~1 s).

Re-running any single model (e.g. `python probe_bench.py --force resnet18_in1k@64`, ~20 s) reproduces its stored accuracy exactly.

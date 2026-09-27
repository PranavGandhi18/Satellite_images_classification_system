# Model Selection: Summary

*Two-page companion to `DESIGN_NOTE.md`. The full evidence is in `model_selection_thoughtprocess.md`, and the code plus raw results are in `experiments/`.*

**Bottom line.**
- **Pick:** a **ViT-S/16 transformer fine-tuned at 112×112 px**, served as an 87 MB ONNX model. It runs at ~30 ms per tile on one CPU thread and scores **98.9%** on held-out tiles.
- **Weights:** start from **DINOv3** if its licence clears; otherwise from Apache-2.0 **ImageNet-21k ViT-S/16** (98.1%, same code).
- **How I chose:** I tested 58 model configurations on this data, on CPU, before choosing.

## 1. Where the model sits in the system

- **The model is a versioned artifact, not code.** `models/<version>/` holds `model.onnx` and `manifest.json`. The manifest records:
  - the class order;
  - the **preprocessing** (64 → 112 px bicubic resize, normalisation);
  - the weights checksum and training-data hash;
  - validation metrics and the threshold policy.
- **The service sees a narrow interface: `image → 7 probabilities + embedding`.** Swapping DINOv3 ↔ ViT-S ↔ ResNet is a new directory, not a code change.
- **Tile flow:** validate → preprocess (from the manifest) → ONNX Runtime on CPU → probabilities + 384-d embedding → confidence, margin and "unfamiliar" scores → stored together with the model version.
- **Why preprocessing lives inside the artifact.** It was the most dangerous thing I found:
  - one downloaded EuroSAT model scores **99% with its (undocumented) input format and 13% with standard preprocessing**;
  - another loses 5 points with a plain resize.
  - Either way there's no error, just confident nonsense.

## 2. Options considered, and how I chose

**Criteria, set before looking at results:**
- runs offline on CPU;
- honest evaluation (it must not have seen our eval tiles);
- useful confidence and embeddings;
- a licence usable in a commercial product;
- retrainable on the isolated box.

| Option | Training | Best measured | Verdict |
|---|---|---|---|
| Zero-shot vision-language (CLIP, SigLIP2, remote-sensing CLIPs) | none | 71% | Too weak as the classifier |
| Specialized checkpoints from Hugging Face (19 tried) | none | 97.7–100% on tiles *they* held out | Accurate, but not trustworthy as-is: **81–91% of our eval tiles were in their training data**, preprocessing is undocumented, the best have **no licence**, and they carry EuroSAT's 10 classes. Models trained on aerial datasets (RESISC45, GID) reach only 29–73%. **Keep one beside production as a second opinion, not serve it.** |
| Frozen pretrained backbone + linear head (31 configurations) | seconds | 97.6% eval (98.3% CV) | Good fallback: can be retrained on the box in seconds |
| **Full fine-tune** (8 models) | ~25 min on CPU | **97.0–98.9%** | **Chosen:** best accuracy, and we own data, labels and calibration |

**Is training necessary?** Pretrained backbones output features, not our labels, so a small amount of training is unavoidable. Fine-tuning is cheap enough on CPU to be part of normal operations: analysts' corrections can be turned into a new model overnight, offline.

**CNN vs transformer: what the measurements say.**
- **Frozen features:** transformers win *when their pretraining is bigger* (ImageNet-21k ViT-S 97.1% CV vs ImageNet ResNet-18 93.8%). But a ResNet-18 pretrained on Sentinel-2 itself matches them (97.1%). **Pretraining data matters more than architecture.**
- **After fine-tuning, architecture stops mattering.** DINOv3 ViT-S, ViT-S, ResNet-18 and SSL4EO ResNet-18 all score 98.1–98.9%, a spread of ≤ 3 tiles out of 368, which is within noise.
- **"Transformers are too slow on CPU" doesn't hold at this input size.** At 112 px (49 patches), attention is ~2% of the compute. ViT-S runs at **29 ms / 1 thread** vs **41 ms** for ResNet-18 @224, and both do ~100 tiles/s on 4 threads. At 224 px the ViT *is* 3.5× slower for no gain, so resolution was the real decision.

| Finalist (fine-tuned) | Accuracy (n=368) | ONNX ms, 1 / 4 threads | Tiles/s | Size | Licence |
|---|---|---|---|---|---|
| **DINOv3 ViT-S/16 @112** | **98.9** | 30.7 / 14.4 | 99 | 87 MB | DINOv3 (custom) |
| ViT-S/16 IN-21k @112 | 98.1 | 28.6 / 13.3 | 109 | 87 MB | Apache-2.0 |
| ResNet-18 @224 | 98.1 | 41.0 / 10.9 | 104 | 45 MB | BSD-3 |
| SSL4EO ResNet-18 @224 | 98.1 | 40.2 / 10.2 | 105 | 45 MB | CC-BY-4.0 |
| EfficientViT-B0 @224 | 97.0 | 4.0 / 3.9 | 352 | 9 MB | Apache-2.0 |

**How I broke the tie.** Accuracy and speed are tied, so the pick rests on:
- **Frozen features at 112 px:** DINOv3's are better (95.8% vs 93.3% CV). That matters for the "unfamiliar input" score and the retrain-in-seconds fallback.
- **Single-thread speed:** it's among the fastest of the top group.
- **Future inputs:** most multi-sensor Earth-observation foundation models, including the SAR + optical ones GalaxEye would move to (CROMA, DOFA, TerraMind…), are ViTs.
- **Licence risk:** it's handled by the fallback, which is the same architecture.

**Alternatives I'd accept:**
- **SSL4EO ResNet-18** (CC-BY-4.0): if a CNN is preferred. Equally accurate, half the size.
- **EfficientViT-B0:** if the hardware is weak. 4 ms per tile, 1.4 min per Sentinel-2 scene (vs ~5 min), at −1.9 points.

## 3. What this choice means for uncertainty, storage and querying

- **Low-confidence predictions:** after fine-tuning, the **80% most-confident predictions are 99.4–100% correct**, so a confidence threshold is a real lever between coverage and accuracy.
  - Pick the threshold on the validation split, never on eval.
  - Store all 7 probabilities, so changing the threshold later needs no re-run.
- **Unfamiliar inputs** (clouds, land types outside the 7): softmax can't flag these, so I use distance in embedding space instead. It's still unvalidated; next experiment: hold one class out as "unknown" and measure how well each finalist detects it.
- **What to store per tile:**
  - the 7 probabilities;
  - the 384-float embedding (~1.5 KB);
  - model version and preprocessing version;
  - the raw tile, so a model upgrade can reprocess history.
- **Querying:**
  - embeddings make "find tiles like this one" cheap;
  - class counts should be corrected with the validation confusion matrix, because noisy labels bias totals.

## 4. Assumptions

- Inputs stay 64×64 RGB Sentinel-2 tiles.
- The 7 classes are fixed for now.
- The CPU is roughly comparable to a modern x86 server core; ARM could reorder the finalists.
- There is no hard latency target, but scene-scale throughput matters (one granule is 29,241 tiles, ~5 min on 4 threads).
- Licences can be reviewed before shipping.
- The balanced eval set is only a proxy for the real class mix.

## 5. Questions I'd ask you

1. **Target hardware and volume:** cores, RAM, ARM or x86; tiles or scenes per day? This decides ViT-S vs EfficientViT.
2. **Licences:** may we ship DINOv3-licensed or CC-BY weights? This decides DINOv3 vs the Apache fallback.
3. **Future inputs:** will real inputs become multispectral or SAR? That changes the backbone family: CROMA, DOFA or an SSL4EO variant.
4. **Classes:** is the taxonomy fixed? What should happen to clouds or land that is none of the 7?
5. **Error costs:** which mistakes are expensive? This sets the confidence threshold, maybe per class.
6. **Labelled data:** can we get a few thousand labelled GalaxEye tiles? Our 210 + 158 test tiles can't separate the tied finalists, and EuroSAT isn't GalaxEye's imagery.
7. **A second model:** may we ship one as a monitor, i.e. a specialized checkpoint whose disagreement flags tiles for review?

*Caveats: small test sets (±2–3 pp), one fine-tuning recipe for all models, and latency measured on one AMD EPYC. Details are in the full note, §8.*

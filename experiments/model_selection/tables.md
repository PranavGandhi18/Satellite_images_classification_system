<!-- PROBE -->
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

<!-- ZS -->
| Model | Zero-shot, 7-way (no training) | Same encoder + linear probe |
|---|---|---|
| GeoRSCLIP ViT-B/32 | 71.4 | 97.6 |
| SigLIP2 ViT-B/16 | 65.2 | 97.6 |
| CLIP ViT-B/32 | 54.8 | 94.8 |
| RemoteCLIP ViT-B/32 | 45.2 | 92.9 |

<!-- FT -->
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

<!-- SPEC -->
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

<!-- COST -->
| Candidate (fine-tuned) | Val+eval acc (n=368) | Frozen-feature CV | Params (M) | GMACs | ONNX size (MB) | ONNX ms, 1 thread | ONNX ms, 4 threads | Tiles/s (4 threads, batch 32) | PyTorch ms, 1 thread | Licence |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT-S/16 (IN-21k) @112 | 98.1 | 93.3 | 21.7 | 1.08 | 87 | 28.6 | 13.3 | 109 | 44.0 | Apache-2.0 |
| DINOv3 ViT-S/16 @112 | 98.9 | 95.8 | 21.6 | 1.16 | 87 | 30.7 | 14.4 | 99 | 49.1 | DINOv3 License |
| ViT-S/16 (IN-21k) @224 | 98.4 | 97.1 | 21.7 | 4.24 | 87 | 104.9 | 48.2 | 31 | 139.3 | Apache-2.0 |
| ResNet-18 @224 | 98.1 | 93.8 | 11.2 | 1.81 | 45 | 41.0 | 10.9 | 104 | 53.0 | BSD-3 |
| SSL4EO ResNet-18 @224 | 98.1 | 97.1 | 11.2 | 1.81 | 45 | 40.2 | 10.2 | 105 | 53.9 | CC-BY-4.0 |
| EfficientViT-B0 @224 | 97.0 | 94.8 | 2.1 | 0.10 | 9 | 4.0 | 3.9 | 352 | 11.0 | Apache-2.0 |
| ResNet-18 @64 | 95.1 | 90.5 | 11.2 | 0.15 | 45 | 7.1 | 1.3 | 899 | 17.7 | BSD-3 |


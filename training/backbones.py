"""Pretrained backbones train.py can start from, and where their vendored weights live.

The weights are downloaded once by scripts/fetch_offline_assets.py (the only step that needs internet) into
weights/<timm_name>/, so training - like the service - runs with no network access.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / 'weights'

# key -> (timm model name, backbone learning rate, licence of the pretrained weights)
BACKBONES = {
    'dinov3_vits16': ('vit_small_patch16_dinov3.lvd1689m', 5e-5, 'DINOv3 License (custom; needs legal review before shipping)'),
    'vit_s16_in21k': ('vit_small_patch16_224.augreg_in21k_ft_in1k', 5e-5, 'Apache-2.0'),
}


def weights_path(timm_name: str) -> Path:
    return WEIGHTS_DIR / timm_name / 'model.safetensors'

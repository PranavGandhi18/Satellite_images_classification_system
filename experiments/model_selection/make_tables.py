"""Build the markdown tables for model_selection_thoughtprocess.md straight from the result JSONs."""
import json, glob, os, math
S = os.path.dirname(os.path.abspath(__file__))
L = lambda d: {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f'{S}/{d}/*.json'))}  # sorted: deterministic tie order
probe, lat, ft, spec = L('feats'), L('latency'), L('ft'), L('spec')

def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h
pct = lambda x: f'{100 * x:.1f}'
def ms(key, which='onnx'):
    r = lat.get(key, {})
    a, b = r.get(f'{which}_ms_1thr'), r.get(f'{which}_ms_4thr')
    return (f'{a:.1f}' if a else '–'), (f'{b:.1f}' if b else '–')

NAMES = {
  'resnet18_in1k': ('ResNet-18', 'CNN', 'ImageNet-1k supervised', 'BSD-3'),
  'resnet18_in1k@64': ('ResNet-18 @64 px', 'CNN', 'ImageNet-1k supervised', 'BSD-3'),
  'resnet50_in1k': ('ResNet-50', 'CNN', 'ImageNet-1k supervised', 'Apache-2.0'),
  'efficientnet_b0': ('EfficientNet-B0', 'CNN', 'ImageNet-1k supervised', 'Apache-2.0'),
  'mobilenetv3_large': ('MobileNetV3-L', 'CNN', 'ImageNet-1k supervised', 'Apache-2.0'),
  'convnext_tiny_in22k': ('ConvNeXt-T', 'CNN', 'ImageNet-22k supervised', 'Apache-2.0'),
  'vit_small16_in21k': ('ViT-S/16', 'Transformer', 'ImageNet-21k supervised', 'Apache-2.0'),
  'vit_small16_in21k@112': ('ViT-S/16 @112 px', 'Transformer', 'ImageNet-21k supervised', 'Apache-2.0'),
  'deit_tiny16': ('DeiT-Tiny', 'Transformer', 'ImageNet-1k supervised', 'Apache-2.0'),
  'swin_tiny_in22k': ('Swin-T', 'Transformer', 'ImageNet-22k supervised', 'MIT'),
  'tiny_vit_5m': ('TinyViT-5M', 'Hybrid', 'ImageNet-22k distilled', 'Apache-2.0'),
  'fastvit_t8': ('FastViT-T8', 'Hybrid', 'ImageNet-1k supervised', 'Apple (custom)'),
  'efficientvit_b0': ('EfficientViT-B0', 'Hybrid', 'ImageNet-1k supervised', 'Apache-2.0'),
  'mobilevitv2_100': ('MobileViTv2-1.0 @256 px', 'Hybrid', 'ImageNet-1k supervised', 'Apple (custom)'),
  'dinov2_vits14': ('DINOv2 ViT-S/14', 'Transformer', 'Self-supervised, LVD-142M', 'Apache-2.0'),
  'dinov2_vits14@112': ('DINOv2 ViT-S/14 @112 px', 'Transformer', 'Self-supervised, LVD-142M', 'Apache-2.0'),
  'dinov2_vitb14': ('DINOv2 ViT-B/14', 'Transformer', 'Self-supervised, LVD-142M', 'Apache-2.0'),
  'dinov3_vits16': ('DINOv3 ViT-S/16', 'Transformer', 'Self-supervised, LVD-1689M', 'DINOv3 License'),
  'dinov3_vits16@112': ('DINOv3 ViT-S/16 @112 px', 'Transformer', 'Self-supervised, LVD-1689M', 'DINOv3 License'),
  'dinov3_vits16@64': ('DINOv3 ViT-S/16 @64 px', 'Transformer', 'Self-supervised, LVD-1689M', 'DINOv3 License'),
  'dinov3_vitb16': ('DINOv3 ViT-B/16', 'Transformer', 'Self-supervised, LVD-1689M', 'DINOv3 License'),
  'dinov3_convnext_tiny': ('DINOv3 ConvNeXt-T', 'CNN', 'Distilled from DINOv3 7B', 'DINOv3 License'),
  'dinov3_vitl16_sat': ('DINOv3-SAT ViT-L/16', 'Transformer', 'Self-supervised, 493M Maxar tiles (0.6 m)', 'DINOv3 License'),
  'ssl4eo_resnet18_rgb_moco': ('SSL4EO-S12 ResNet-18 (RGB)', 'CNN', 'MoCo on 1M Sentinel-2 patches', 'CC-BY-4.0'),
  'ssl4eo_resnet50_rgb_moco': ('SSL4EO-S12 ResNet-50 (RGB)', 'CNN', 'MoCo on 1M Sentinel-2 patches', 'CC-BY-4.0'),
  'satlas_resnet50_s2_rgb': ('SatlasPretrain ResNet-50 (S2 RGB)', 'CNN', 'Supervised multi-task, Sentinel-2', 'ODC-BY'),
  'satlas_swinv2t_s2_rgb': ('SatlasPretrain Swin-v2-T (S2 RGB)', 'Transformer', 'Supervised multi-task, Sentinel-2', 'ODC-BY'),
  'clip_vitb32_openai': ('CLIP ViT-B/32', 'Transformer', 'Image-text, 400M pairs', 'MIT'),
  'clip_vitb16_siglip2': ('SigLIP2 ViT-B/16', 'Transformer', 'Image-text, WebLI', 'Apache-2.0'),
  'remoteclip_vitb32': ('RemoteCLIP ViT-B/32', 'Transformer', 'CLIP + remote-sensing captions', 'Apache-2.0'),
  'georsclip_vitb32': ('GeoRSCLIP ViT-B/32', 'Transformer', 'CLIP + RS5M captions', 'repo MIT, weights "cc"'),
}
VISION_PARAMS = {'clip_vitb32_openai': 87.8, 'remoteclip_vitb32': 87.8, 'georsclip_vitb32': 87.8, 'clip_vitb16_siglip2': 92.9}

def probe_table():
    rows = sorted(probe.values(), key=lambda r: (-r['cv_acc'], r['key']))
    out = ['| # | Backbone | Type | Pretraining | Input | Params (M) | GMACs | LP CV acc (n=1,050) | LP eval acc (n=210) | ONNX ms, 1 / 4 threads | Licence |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows, 1):
        k = r['key']; n, fam, pre, lic = NAMES[k]
        g = r['gflops'] / 2 if r['gflops'] == r['gflops'] else float('nan')
        a, b = ms(k)
        p = VISION_PARAMS.get(k, r['params_m'])
        out.append(f"| {i} | {n} | {fam} | {pre} | {r['input']} | {p:.1f} | {'–' if g != g else f'{g:.2f}'} | {pct(r['cv_acc'])} | {pct(r['eval_acc'])} | {a} / {b} | {lic} |")
    return '\n'.join(out)

def zs_table():
    rows = sorted([r for r in probe.values() if 'zeroshot_acc' in r], key=lambda r: (-r['zeroshot_acc'], r['key']))
    out = ['| Model | Zero-shot, 7-way (no training) | Same encoder + linear probe |', '|---|---|---|']
    for r in rows: out.append(f"| {NAMES[r['key']][0]} | {pct(r['zeroshot_acc'])} | {pct(r['eval_acc'])} |")
    return '\n'.join(out)

FTN = {'resnet18.tv_in1k': ('ResNet-18', 'CNN', 'resnet18_in1k'), 'efficientvit_b0.r224_in1k': ('EfficientViT-B0', 'Hybrid', 'efficientvit_b0'),
       'mobilenetv3_large_100.ra_in1k': ('MobileNetV3-L', 'CNN', 'mobilenetv3_large'), 'vit_small_patch16_224.augreg_in21k_ft_in1k': ('ViT-S/16 (IN-21k)', 'Transformer', 'vit_small16_in21k'),
       'vit_small_patch16_dinov3.lvd1689m': ('DINOv3 ViT-S/16', 'Transformer', 'dinov3_vits16'), 'ssl4eo_resnet18_rgb_moco': ('SSL4EO ResNet-18 (Sentinel-2)', 'CNN', 'ssl4eo_resnet18_rgb_moco')}
def ft_table():
    out = ['| Fine-tuned model | Type | Input | LP eval → **FT eval** (n=210) | FT val (n=158) | Val+eval correct (n=368) | 95% CI | Acc. on 80% most-confident | CPU train time (threads) | ONNX ms, 1 / 4 threads |',
           '|---|---|---|---|---|---|---|---|---|---|']
    rows = []
    for r in ft.values():
        n, fam, pk = FTN[r['model']]
        pk2 = pk if r['input'] == 224 else f"{pk}@{r['input']}"
        lp = probe.get(pk2, {}).get('eval_acc')
        k = round(r['val_acc'] * 158) + round(r['eval_acc'] * 210); lo, hi = wilson(k, 368)
        a, b = ms(pk2)
        rows.append(((-k, n, r['input']), f"| {n} | {fam} | {r['input']} | {pct(lp) if lp else '–'} → **{pct(r['eval_acc'])}** | {pct(r['val_acc'])} | {k}/368 = **{pct(k / 368)}** | {pct(lo)}–{pct(hi)} | {pct(r['acc_at_80cov'])} | {r['train_minutes_cpu']:.0f} min ({r['threads']}) | {a} / {b} |"))
    return '\n'.join(out + [r for _, r in sorted(rows)])

SPN = {
  'cm93_resnet18': ('ResNet-18', 'CNN'), 'cm93_resnet50': ('ResNet-50', 'CNN'), 'huygens_resnet50': ('ResNet-50', 'CNN'),
  'taufiqdp_convnextv2_t': ('ConvNeXt-V2-T @384', 'CNN'), 'bknyaz_vit_l16': ('ViT-L/16', 'Transformer'), 'mrm8488_convnext_t': ('ConvNeXt-T', 'CNN'),
  'nielsr_swin_t': ('Swin-T', 'Transformer'), 'nielsr_vit_b16': ('ViT-B/16', 'Transformer'), 'nielsr_van_b': ('VAN-B', 'CNN (large-kernel attention)'),
  'adilbai_swin': ('Swin', 'Transformer'), 'tanganke_convnext_b': ('ConvNeXt-B', 'CNN'), 'tanganke_clip_b32': ('CLIP ViT-B/32 (fine-tuned)', 'Transformer'),
  'tanganke_clip_b16': ('CLIP ViT-B/16 (fine-tuned)', 'Transformer'), 'resisc_clip_b32': ('CLIP ViT-B/32 (fine-tuned)', 'Transformer'),
  'resisc_convnext_b': ('ConvNeXt-B', 'CNN'), 'resisc_siglip2': ('SigLIP2-B', 'Transformer'), 'gid_siglip2': ('SigLIP2-B', 'Transformer'),
  'rsscn7_vit': ('ViT-B/16', 'Transformer'), 'ucm_clip': ('CLIP ViT-B/32 (fine-tuned), zero-shot head', 'Transformer'),
}
def spec_table():
    out = ['| HF repo | Arch | Trained on | Covers our classes | Acc. eval (n=210) | Acc. all 1,260 | Acc. with plain resize | Top-1 is a usable label | Held-out estimate† | torch ms, 1 / 4 threads |',
           '|---|---|---|---|---|---|---|---|---|---|']
    rows = []
    for k, r in spec.items():
        n, fam = SPN[k]
        plain = r['by_prep'].get('plain', {}).get('acc_all1260')
        held = r.get('acc_on_our_tiles_in_test_split'); held = f"{pct(held[0])} (n={held[1]})" if held else '–'
        a, b = r.get('torch_ms_1thr'), r.get('torch_ms_4thr')
        cov = '7/7' if len(r['covers']) == 7 else f"{len(r['covers'])}/7 (no {', '.join(c for c in ['AnnualCrop','Forest','Highway','Industrial','Residential','River','SeaLake'] if c not in r['covers'])})"
        rows.append(((-r['acc_all1260'], r['repo']), f"| `{r['repo']}` | {n} | {r['trained_on']} | {cov} | {pct(r['acc_eval'])} | {pct(r['acc_all1260'])} | {pct(plain) if plain is not None else '–'} | {pct(r['in_vocab_rate'])} | {held} | {f'{a:.0f}' if a else '–'} / {f'{b:.0f}' if b else '–'} |"))
    return '\n'.join(out + [r for _, r in sorted(rows)])

FINALISTS = [  # (label, probe key for cost/latency, fine-tune file key)
  ('ViT-S/16 (IN-21k) @112', 'vit_small16_in21k@112', 'vit_small_patch16_224.augreg_in21k_ft_in1k@112'),
  ('DINOv3 ViT-S/16 @112', 'dinov3_vits16@112', 'vit_small_patch16_dinov3.lvd1689m@112'),
  ('ViT-S/16 (IN-21k) @224', 'vit_small16_in21k', 'vit_small_patch16_224.augreg_in21k_ft_in1k@224'),
  ('ResNet-18 @224', 'resnet18_in1k', 'resnet18.tv_in1k@224'),
  ('SSL4EO ResNet-18 @224', 'ssl4eo_resnet18_rgb_moco', 'ssl4eo_resnet18_rgb_moco@224'),
  ('EfficientViT-B0 @224', 'efficientvit_b0', 'efficientvit_b0.r224_in1k@224'),
  ('ResNet-18 @64', 'resnet18_in1k@64', 'resnet18.tv_in1k@64'),
]
def cost_table():
    out = ['| Candidate (fine-tuned) | Val+eval acc (n=368) | Frozen-feature CV | Params (M) | GMACs | ONNX size (MB) | ONNX ms, 1 thread | ONNX ms, 4 threads | Tiles/s (4 threads, batch 32) | PyTorch ms, 1 thread | Licence |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for lab, pk, fk in FINALISTS:
        p, l, f = probe.get(pk, {}), lat.get(pk, {}), ft.get(fk, {})
        k = round(f['val_acc'] * 158) + round(f['eval_acc'] * 210) if f else None
        g = p.get('gflops', float('nan')) / 2
        fmt = lambda v, d=1: '–' if v is None else f'{v:.{d}f}'
        out.append(f"| {lab} | {fmt(100 * k / 368) if k else '–'} | {pct(p['cv_acc']) if p else '–'} | {fmt(p.get('params_m'))} | {fmt(g, 2)} | {fmt(l.get('onnx_mb'), 0)} | {fmt(l.get('onnx_ms_1thr'))} | {fmt(l.get('onnx_ms_4thr'))} | {fmt(l.get('onnx_tiles_per_s_4thr_b32'), 0)} | {fmt(l.get('torch_ms_1thr'))} | {NAMES[pk][3]} |")
    return '\n'.join(out)

if __name__ == '__main__':
    for name, fn in [('PROBE', probe_table), ('ZS', zs_table), ('FT', ft_table), ('SPEC', spec_table), ('COST', cost_table)]:
        print(f'<!-- {name} -->'); print(fn()); print()

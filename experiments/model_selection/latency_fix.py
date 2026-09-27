"""Re-run the latency/ONNX pass for ViTs whose export failed, built the way production would build them:
fixed img_size (no runtime pos-embed resampling) and timm's non-fused attention (exportable by the legacy exporter).
Accuracy is unaffected: these only change how the same weights are traced."""
import sys, timm, torch
try: timm.layers.set_fused_attn(False)
except Exception as e: print('set_fused_attn unavailable', e)
sys.argv, keys = sys.argv[:1], sys.argv[1:]
import latency
SPECIAL = {'vit_small16_in21k@112': ('vit_small_patch16_224.augreg_in21k_ft_in1k', 112),
           'dinov3_vits16@112': ('vit_small_patch16_dinov3.lvd1689m', 112),
           'dinov3_vits16@64': ('vit_small_patch16_dinov3.lvd1689m', 64),
           'dinov3_vits16': ('vit_small_patch16_dinov3.lvd1689m', 224)}
latency.make = lambda key: (timm.create_model(SPECIAL[key][0], pretrained=True, num_classes=0, img_size=SPECIAL[key][1]).eval(), SPECIAL[key][1])
for k in keys or SPECIAL:
    try: latency.run(k)
    except Exception as e: print('FAILED', k, e, flush=True)

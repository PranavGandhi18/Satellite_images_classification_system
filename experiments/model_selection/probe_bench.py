"""Frozen-backbone benchmark: linear probe (+ zero-shot for CLIP-style models) and CPU cost.

Features are cached in feats/<key>.npz. Classifier is fit on candidate_tiles only
(C picked by 5-fold CV on candidates), then scored once on eval_set.
"""
import os, sys, glob, json, time, warnings, math
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

R = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'be-mlsys-assignment-dataset'))
S = os.path.dirname(os.path.abspath(__file__))
FE = f'{S}/feats'; os.makedirs(FE, exist_ok=True)
CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']
labels = dict(l.strip().split(',') for l in open(f'{R}/eval_labels.csv').readlines()[1:])
cand = sorted(glob.glob(f'{R}/candidate_tiles/*/*.png')); ev = sorted(glob.glob(f'{R}/eval_set/*.png'))
yc = np.array([CLASSES.index(p.split('/')[-2]) for p in cand]); ye = np.array([CLASSES.index(labels[os.path.basename(p)]) for p in ev])
RAW = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in cand + ev])  # (1260,64,64,3) uint8
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def to_tensor(raw, size, mean, std, scale=1 / 255.):
    x = torch.from_numpy(raw).permute(0, 3, 1, 2).float() * scale
    if size != raw.shape[1]:
        x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

# ---------------------------------------------------------------- model zoo
def timm_model(name, **kw):
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0, **kw).eval()
    cfg = m.pretrained_cfg
    return m, cfg.get('mean', IMNET[0]), cfg.get('std', IMNET[1])

def tg_resnet(arch, wname):
    import timm, torchgeo.models as TG
    W = getattr(getattr(TG, 'ResNet' + arch[len('resnet'):] + '_Weights'), wname)
    m = getattr(TG, arch)(weights=W); m.reset_classifier(0); return m.eval()

def tg_swin_t_satlas():
    import torchgeo.models as TG
    m = TG.swin_v2_t(weights=TG.Swin_V2_T_Weights.SENTINEL2_SI_RGB_SATLAS); m.head = nn.Identity(); return m.eval()

def openclip(arch, pretrained=None, hf=None):
    import open_clip
    from huggingface_hub import hf_hub_download
    m, _, pre = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    if hf:
        sd = torch.load(hf_hub_download(*hf), map_location='cpu', weights_only=False)
        sd = sd.get('state_dict', sd); sd = {k.replace('module.', ''): v for k, v in sd.items()}
        print('  load', m.load_state_dict(sd, strict=False))
    tok = open_clip.get_tokenizer(arch)
    mean = m.visual.preprocess_cfg.get('mean', (0.48145466, 0.4578275, 0.40821073)) if hasattr(m.visual, 'preprocess_cfg') else (0.48145466, 0.4578275, 0.40821073)
    std = m.visual.preprocess_cfg.get('std', (0.26862954, 0.26130258, 0.27577711)) if hasattr(m.visual, 'preprocess_cfg') else (0.26862954, 0.26130258, 0.27577711)
    return m.eval(), tok, mean, std

class CLIPVis(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return s.m.encode_image(x)

# key: (group, family, builder, input size)
ZOO = {
  # ImageNet-supervised CNNs
  'resnet18_in1k':        ('ImageNet CNN', 'CNN', lambda: timm_model('resnet18.tv_in1k'), 224),
  'resnet18_in1k@64':     ('ImageNet CNN', 'CNN', lambda: timm_model('resnet18.tv_in1k'), 64),
  'resnet50_in1k':        ('ImageNet CNN', 'CNN', lambda: timm_model('resnet50.a1_in1k'), 224),
  'efficientnet_b0':      ('ImageNet CNN', 'CNN', lambda: timm_model('efficientnet_b0.ra_in1k'), 224),
  'mobilenetv3_large':    ('ImageNet CNN', 'CNN', lambda: timm_model('mobilenetv3_large_100.ra_in1k'), 224),
  'convnext_tiny_in22k':  ('ImageNet CNN', 'CNN', lambda: timm_model('convnext_tiny.fb_in22k_ft_in1k'), 224),
  # ImageNet-supervised transformers / hybrids
  'vit_small16_in21k':    ('ImageNet Transformer', 'ViT', lambda: timm_model('vit_small_patch16_224.augreg_in21k_ft_in1k'), 224),
  'vit_small16_in21k@112': ('ImageNet Transformer', 'ViT', lambda: timm_model('vit_small_patch16_224.augreg_in21k_ft_in1k', dynamic_img_size=True), 112),
  'deit_tiny16':          ('ImageNet Transformer', 'ViT', lambda: timm_model('deit_tiny_patch16_224.fb_in1k'), 224),
  'swin_tiny_in22k':      ('ImageNet Transformer', 'Swin', lambda: timm_model('swin_tiny_patch4_window7_224.ms_in22k_ft_in1k'), 224),
  'tiny_vit_5m':          ('ImageNet Transformer', 'Hybrid', lambda: timm_model('tiny_vit_5m_224.dist_in22k_ft_in1k'), 224),
  'fastvit_t8':           ('ImageNet Transformer', 'Hybrid', lambda: timm_model('fastvit_t8.apple_in1k'), 224),
  'efficientvit_b0':      ('ImageNet Transformer', 'Hybrid', lambda: timm_model('efficientvit_b0.r224_in1k'), 224),
  'mobilevitv2_100':      ('ImageNet Transformer', 'Hybrid', lambda: timm_model('mobilevitv2_100.cvnets_in1k'), 256),
  # Self-supervised general-purpose
  'dinov2_vits14':        ('Self-supervised (general)', 'ViT', lambda: timm_model('vit_small_patch14_dinov2.lvd142m', img_size=224), 224),
  'dinov2_vits14@112':    ('Self-supervised (general)', 'ViT', lambda: timm_model('vit_small_patch14_dinov2.lvd142m', dynamic_img_size=True), 112),
  'dinov2_vitb14':        ('Self-supervised (general)', 'ViT', lambda: timm_model('vit_base_patch14_dinov2.lvd142m', img_size=224), 224),
  'dinov3_vits16':        ('Self-supervised (general)', 'ViT', lambda: timm_model('vit_small_patch16_dinov3.lvd1689m'), 224),
  'dinov3_vits16@112':    ('Self-supervised (general)', 'ViT', lambda: timm_model('vit_small_patch16_dinov3.lvd1689m', dynamic_img_size=True), 112),
  'dinov3_vits16@64':     ('Self-supervised (general)', 'ViT', lambda: timm_model('vit_small_patch16_dinov3.lvd1689m', dynamic_img_size=True), 64),
  'dinov3_vitb16':        ('Self-supervised (general)', 'ViT', lambda: timm_model('vit_base_patch16_dinov3.lvd1689m'), 224),
  'dinov3_convnext_tiny': ('Self-supervised (general)', 'CNN', lambda: timm_model('convnext_tiny.dinov3_lvd1689m'), 224),
  # Remote-sensing pretrained
  'dinov3_vitl16_sat':    ('Remote-sensing pretrained', 'ViT', lambda: timm_model('vit_large_patch16_dinov3.sat493m'), 224),
  'ssl4eo_resnet18_rgb_moco': ('Remote-sensing pretrained', 'CNN', 'tg:resnet18:SENTINEL2_RGB_MOCO', None),
  'ssl4eo_resnet50_rgb_moco': ('Remote-sensing pretrained', 'CNN', 'tg:resnet50:SENTINEL2_RGB_MOCO', None),
  'satlas_resnet50_s2_rgb':   ('Remote-sensing pretrained', 'CNN', 'tg:resnet50:SENTINEL2_SI_RGB_SATLAS', None),
  'satlas_swinv2t_s2_rgb':    ('Remote-sensing pretrained', 'Swin', 'tg:swin_t', None),
  # Vision-language (linear probe + zero-shot)
  'clip_vitb32_openai':   ('Vision-language', 'ViT', ('clip', 'ViT-B-32', 'openai', None), 224),
  'clip_vitb16_siglip2':  ('Vision-language', 'ViT', ('clip', 'ViT-B-16-SigLIP2', 'webli', None), 224),
  'remoteclip_vitb32':    ('Vision-language (RS)', 'ViT', ('clip', 'ViT-B-32', None, ('chendelong/RemoteCLIP', 'RemoteCLIP-ViT-B-32.pt')), 224),
  'georsclip_vitb32':     ('Vision-language (RS)', 'ViT', ('clip', 'ViT-B-32', 'openai', ('Zilun/GeoRSCLIP', 'ckpt/RS5M_ViT-B-32.pt')), 224),
}

# zero-shot class names (following the CLIP paper's EuroSAT names)
ZS_NAMES = {'AnnualCrop': 'annual crop land', 'Forest': 'forest', 'Highway': 'highway or road',
            'Industrial': 'industrial buildings or commercial buildings', 'Residential': 'residential buildings or homes or apartments',
            'River': 'river', 'SeaLake': 'lake or sea'}
ZS_TEMPLATES = ['a satellite photo of {}.', 'a centered satellite photo of {}.', 'an aerial photo of {}.', 'satellite imagery of {}.', 'a land use image of {}.']

def cost(model, x1):
    from torch.utils.flop_counter import FlopCounterMode
    params = sum(p.numel() for p in model.parameters()) / 1e6
    try:
        with torch.no_grad():
            fc = FlopCounterMode(display=False)
            with fc: model(x1)
            gflops = fc.get_total_flops() / 1e9
    except Exception as e:
        print('  flop count failed:', e); gflops = float('nan')
    lat = {}
    for th in (1, 4):
        torch.set_num_threads(th)
        with torch.inference_mode():
            for _ in range(3): model(x1)
            n = 5 if gflops > 50 else 20; ts = []
            for _ in range(n):
                t = time.perf_counter(); model(x1); ts.append((time.perf_counter() - t) * 1000)
        lat[th] = float(np.median(ts))
    torch.set_num_threads(64)
    return params, gflops, lat

def extract(model, mk_input, bs=64):
    out = []
    with torch.inference_mode():
        for i in range(0, len(RAW), bs):
            out.append(model(mk_input(RAW[i:i + bs])).float().reshape(min(bs, len(RAW) - i), -1))
    return torch.cat(out).numpy()

def build(key):
    group, fam, b, size = ZOO[key]
    zs = None
    if isinstance(b, str) and b.startswith('tg:'):
        _, arch, *w = b.split(':')
        if arch == 'swin_t':
            m = tg_swin_t_satlas(); scales = [1 / 255.]; size_opts = [64, 256]; mean, std = (0, 0, 0), (1, 1, 1)
        else:
            m = tg_resnet(arch, w[0])
            if 'MOCO' in w[0]:  # expects reflectance DN / 1e4; 8-bit PNG radiometry unknown -> CV-select the scale
                mean, std = (0, 0, 0), (1, 1, 1); scales = [1 / 255., 0.3 / 255., 0.15 / 255.]
            else:  # Satlas: uint8 / 255
                mean, std = (0, 0, 0), (1, 1, 1); scales = [1 / 255.]
            size_opts = [64, 224]
        # choose (scale,size) by 5-fold CV on candidates only
        best = None
        for sc in scales:
            for sz in size_opts:
                f = extract(m, lambda r, sz=sz, sc=sc: to_tensor(r, sz, mean, std, sc))
                cv = probe_cv(f[:len(cand)])[0]
                print(f'   {key} scale={sc:.5f} size={sz} cv={cv:.3f}')
                if best is None or cv > best[0]: best = (cv, sc, sz, f)
        _, sc, size, feats = best
        mk = lambda r: to_tensor(r, size, mean, std, sc)
        return m, mk, feats, size, None
    if isinstance(b, tuple) and b[0] == 'clip':
        _, arch, pre, hf = b
        cm, tok, mean, std = openclip(arch, pre, hf)
        m = CLIPVis(cm).eval()
        mk = lambda r: to_tensor(r, size, mean, std)
        feats = extract(m, mk)
        with torch.inference_mode():
            W = []
            for c in CLASSES:
                t = cm.encode_text(tok([tp.format(ZS_NAMES[c]) for tp in ZS_TEMPLATES])); t = F.normalize(t, dim=-1).mean(0)
                W.append(F.normalize(t, dim=0))
            W = torch.stack(W).numpy()
        zs = W
        return m, mk, feats, size, zs
    m, mean, std = b()
    mk = lambda r: to_tensor(r, size, mean, std)
    return m, mk, extract(m, mk), size, None

def probe_cv(Xc):
    best = (-1, None)
    for C in (0.01, 0.1, 1.0):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        cv = cross_val_score(clf, Xc, yc, cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=5).mean()
        if cv > best[0]: best = (cv, C)
    return best

def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

def run(key):
    out = f'{FE}/{key}.json'
    if os.path.exists(out): return json.load(open(out))
    t0 = time.time(); print('==', key, flush=True)
    torch.set_num_threads(64)
    m, mk, feats, size, zs = build(key)
    Xc, Xe = feats[:len(cand)], feats[len(cand):]
    cv, C = probe_cv(Xc)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xc, yc)
    P = clf.predict_proba(Xe); pr = P.argmax(1); acc = float((pr == ye).mean())
    conf = P.max(1); order = np.argsort(-conf); keep = order[:int(0.8 * len(ye))]
    res = dict(key=key, group=ZOO[key][0], family=ZOO[key][1], input=size, dim=int(feats.shape[1]), C=C,
               cv_acc=float(cv), eval_acc=acc, eval_ci=wilson(acc, len(ye)), eval_macro_f1=float(f1_score(ye, pr, average='macro')),
               acc_at_80cov=float((pr[keep] == ye[keep]).mean()),
               per_class_recall={c: float((pr[ye == i] == i).mean()) for i, c in enumerate(CLASSES)})
    if zs is not None:
        fe = Xe / np.linalg.norm(Xe, axis=1, keepdims=True); res['zeroshot_acc'] = float(((fe @ zs.T).argmax(1) == ye).mean())
    params, gflops, lat = cost(m, mk(RAW[:1]))
    res.update(params_m=params, gflops=gflops, cpu_ms_1thr=lat[1], cpu_ms_4thr=lat[4], wall_s=time.time() - t0)
    np.savez_compressed(f'{FE}/{key}.npz', feats=feats.astype(np.float16), proba_eval=P)
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ['key', 'cv_acc', 'eval_acc', 'params_m', 'gflops', 'cpu_ms_1thr', 'cpu_ms_4thr'] }), res.get('zeroshot_acc', ''), flush=True)
    return res

if __name__ == '__main__':
    args = sys.argv[1:]; force = '--force' in args  # --force: recompute even if feats/<key>.json exists
    keys = [a for a in args if a != '--force'] or list(ZOO)
    for k in keys:
        if force and os.path.exists(f'{FE}/{k}.json'): os.remove(f'{FE}/{k}.json')
        try: run(k)
        except Exception as e:
            import traceback; traceback.print_exc(); print('FAILED', k, e, flush=True)

"""Specialized, already-trained satellite land-use classifiers, used as-is (zero training by us).

For each model: map its label set onto our 7 classes (sum of probabilities of the source labels that map to each
of our classes; source labels with no counterpart are "unmapped"), then score on all 1,260 tiles (none were used
by us to train these). Also records: how often the unrestricted top-1 is a label we can use ("in-vocabulary rate")
and the mean probability mass on unmapped labels.
usage: python specialized.py [keys...]            -> accuracy runs, results in spec/<key>.json
       python specialized.py --latency [keys...]  -> CPU latency only (run on an idle machine)
"""
import os, sys, glob, json, re, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
R = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'be-mlsys-assignment-dataset'))
S = os.path.dirname(os.path.abspath(__file__)); OUT = f'{S}/spec'; os.makedirs(OUT, exist_ok=True)
CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']
labels = dict(l.strip().split(',') for l in open(f'{R}/eval_labels.csv').readlines()[1:])
paths = sorted(glob.glob(f'{R}/candidate_tiles/*/*.png')) + sorted(glob.glob(f'{R}/eval_set/*.png'))
names = [('eval/' if 'eval_set' in p else 'cand/') + os.path.basename(p) for p in paths]
y = np.array([CLASSES.index(p.split('/')[-2] if 'candidate' in p else labels[os.path.basename(p)]) for p in paths])
is_eval = np.array([n.startswith('eval') for n in names])
RAW = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in paths])
norm = lambda s: re.sub(r'[^a-z]', '', s.lower())

# source-label -> our class (normalised names). Anything not listed is "unmapped".
MAP = {
  # EuroSAT spellings
  'annualcrop': 'AnnualCrop', 'annualcropland': 'AnnualCrop', 'forest': 'Forest', 'highway': 'Highway', 'highwayorroad': 'Highway',
  'industrial': 'Industrial', 'industrialbuildingsorcommercialbuildings': 'Industrial', 'residential': 'Residential',
  'residentialbuildingsorhomesorapartments': 'Residential', 'river': 'River', 'sealake': 'SeaLake', 'seaorlake': 'SeaLake', 'lakeorsea': 'SeaLake',
  # RESISC45
  'rectangularfarmland': 'AnnualCrop', 'circularfarmland': 'AnnualCrop', 'freeway': 'Highway', 'industrialarea': 'Industrial',
  'denseresidential': 'Residential', 'mediumresidential': 'Residential', 'sparseresidential': 'Residential', 'lake': 'SeaLake',
  # GID-15
  'drycropland': 'AnnualCrop', 'irrigatedland': 'AnnualCrop', 'paddyfield': 'AnnualCrop', 'arborwoodland': 'Forest', 'trafficland': 'Highway',
  'industrialland': 'Industrial', 'ruralresidential': 'Residential', 'urbanresidential': 'Residential', 'pond': 'SeaLake',
  # RSSCN7 ('river or lake' is ambiguous -> maps to both; ties are broken towards River)
  'field': 'AnnualCrop', 'industry': 'Industrial', 'resident': 'Residential', 'riverorlake': ('River', 'SeaLake'),
}

RESISC = ['airplane', 'airport', 'baseball diamond', 'basketball court', 'beach', 'bridge', 'chaparral', 'church', 'circular farmland', 'cloud',
          'commercial area', 'dense residential', 'desert', 'forest', 'freeway', 'golf course', 'ground track field', 'harbor', 'industrial area',
          'intersection', 'island', 'lake', 'meadow', 'medium residential', 'mobile home park', 'mountain', 'overpass', 'palace', 'parking lot',
          'railway', 'railway station', 'rectangular farmland', 'river', 'roundabout', 'runway', 'sea ice', 'ship', 'snowberg', 'sparse residential',
          'stadium', 'storage tank', 'tennis court', 'terrace', 'thermal power station', 'wetland']
EUROSAT_CLIP = ['annual crop land', 'forest', 'brushland or shrubland', 'highway or road', 'industrial buildings or commercial buildings',
                'pasture land', 'permanent crop land', 'residential buildings or homes or apartments', 'river', 'lake or sea']
OUR_CLIP = ['annual crop land', 'forest', 'highway or road', 'industrial buildings or commercial buildings',
            'residential buildings or homes or apartments', 'river', 'lake or sea']
TEMPLATES = ['a centered satellite photo of {}.', 'a satellite photo of {}.', 'satellite imagery of {}.', 'an aerial photo of {}.']

# key: (kind, repo, extra, training data / note)
ZOO = {
  'cm93_resnet18':        ('timm', 'cm93/resnet18-eurosat', None, 'EuroSAT (cm93/eurosat 80/10/10)'),
  'cm93_resnet50':        ('timm', 'cm93/resnet50-eurosat', None, 'EuroSAT (cm93/eurosat 80/10/10)'),
  'huygens_resnet50':     ('timm_pth', 'huygens-jnr/eurosat-resnet50', None, 'EuroSAT (split unknown)'),
  'taufiqdp_convnextv2_t':('timm', 'taufiqdp/convnext-eurosat', None, 'EuroSAT (split unknown)'),
  'bknyaz_vit_l16':       ('timm_labels', 'bknyaz/vitl-eurosat-in21k', 'vit_large_patch16_224.augreg_in21k', 'EuroSAT (jonathan-roberts1, 90/10)'),
  'mrm8488_convnext_t':   ('hf', 'mrm8488/convnext-tiny-finetuned-eurosat', None, 'EuroSAT (nielsr/eurosat-demo, random 90/10)'),
  'nielsr_swin_t':        ('hf', 'nielsr/swin-tiny-patch4-window7-224-finetuned-eurosat', None, 'EuroSAT (nielsr/eurosat-demo, random 90/10)'),
  'nielsr_vit_b16':       ('hf', 'nielsr/vit-finetuned-eurosat-kornia', None, 'EuroSAT (split unknown)'),
  'nielsr_van_b':         ('hf', 'nielsr/van-base-finetuned-eurosat-imgaug', None, 'EuroSAT (split unknown)'),
  'adilbai_swin':         ('hf', 'Adilbai/EuroSAT-Swin', None, 'EuroSAT (nielsr/eurosat-demo)'),
  'tanganke_convnext_b':  ('hf', 'tanganke/convnext-base-224_eurosat_sgd_batch-size-64_lr-0.01_steps-4000', None, 'EuroSAT (tanganke/eurosat)'),
  'tanganke_clip_b32':    ('clipft', 'tanganke/clip-vit-base-patch32_eurosat', ('openai/clip-vit-base-patch32', EUROSAT_CLIP), 'EuroSAT (tanganke/eurosat 21.6k/2.7k)'),
  'tanganke_clip_b16':    ('clipft', 'tanganke/clip-vit-base-patch16_eurosat', ('openai/clip-vit-base-patch16', EUROSAT_CLIP), 'EuroSAT (tanganke/eurosat 21.6k/2.7k)'),
  # other land-use datasets (different sensors / resolutions), mapped onto our classes
  'resisc_clip_b32':      ('clipft', 'tanganke/clip-vit-base-patch32_resisc45', ('openai/clip-vit-base-patch32', RESISC), 'RESISC45 (aerial, 0.2-30 m)'),
  'resisc_convnext_b':    ('hf', 'tanganke/convnext-base-224_resisc45_sgd_batch-size-64_lr-0.01_steps-4000', None, 'RESISC45 (aerial, 0.2-30 m)'),
  'resisc_siglip2':       ('hf', 'prithivMLmods/RESISC45-SigLIP2', None, 'RESISC45 (aerial, 0.2-30 m)'),
  'gid_siglip2':          ('hf', 'prithivMLmods/GiD-Land-Cover-Classification', None, 'GID-15 (Gaofen-2, 4 m)'),
  'rsscn7_vit':           ('hf', 'SeyedAli/Remote-Sensing-UAV-image-classification', None, 'RSSCN7 (aerial)'),
  'ucm_clip':             ('clipfull', 'NemesisAlm/clip-fine-tuned-satellite', OUR_CLIP, 'UC Merced (aerial, 0.3 m); zero-shot with our class names'),
}

_emb = lambda o: o if torch.is_tensor(o) else o.pooler_output  # transformers>=5 returns an output object

def resize_norm(raw, size, mean, std):
    x = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.
    x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False).clamp(0, 1)
    return (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

def hf_prep_cfg(repo):
    from huggingface_hub import hf_hub_download
    try: c = json.load(open(hf_hub_download(repo, 'preprocessor_config.json')))
    except Exception: c = {}
    sz = c.get('size', 224); sz = sz.get('height') or sz.get('shortest_edge') if isinstance(sz, dict) else sz
    return int(sz or 224), c.get('image_mean', [0.485, 0.456, 0.406]), c.get('image_std', [0.229, 0.224, 0.225])

def build(key):
    """returns (net: pixel tensor -> probs over source labels, source label names, model, preps, primary prep name)
    preps: name -> fn(raw uint8 batch) -> pixel tensor. 'native' = the model's own declared pipeline;
    'plain' = generic bicubic resize to the declared size + declared mean/std (what a naive integration would do)."""
    kind, repo, extra, _ = ZOO[key]
    pil = lambda r: [Image.fromarray(a) for a in r]
    if kind in ('timm', 'timm_labels', 'timm_pth'):
        import timm
        from huggingface_hub import hf_hub_download
        if kind == 'timm':
            m = timm.create_model(f'hf-hub:{repo}', pretrained=True)
            c = json.load(open(hf_hub_download(repo, 'config.json'))); src = c['label_names']
        elif kind == 'timm_labels':
            src = [v for k, v in sorted(json.load(open(hf_hub_download(repo, 'labels.json')))['id2label'].items(), key=lambda kv: int(kv[0]))]
            m = timm.create_model(extra, pretrained=False, num_classes=len(src))
            sd = torch.load(hf_hub_download(repo, 'pytorch_model.bin'), map_location='cpu'); print('  load', m.load_state_dict(sd.get('state_dict', sd), strict=False))
        else:
            c = json.load(open(hf_hub_download(repo, 'config.json'))); src = c['class_names']
            m = timm.create_model(c['model_name'], pretrained=False, num_classes=len(src))
            sd = torch.load(hf_hub_download(repo, 'model.pth'), map_location='cpu', weights_only=False)
            sd = sd.get('model_state_dict', sd.get('state_dict', sd)); print('  load', m.load_state_dict(sd, strict=False))
        m.eval(); cfg = m.pretrained_cfg; size = (cfg.get('input_size') or (3, 224, 224))[-1]
        mean, std = cfg.get('mean', (0.485, 0.456, 0.406)), cfg.get('std', (0.229, 0.224, 0.225))
        net = lambda x: F.softmax(m(x), 1)
        preps = {'plain': lambda r: resize_norm(r, size, mean, std)}
        if kind == 'timm_pth':  # no preprocessing documented anywhere: try the obvious alternatives
            preps['raw64_no_norm'] = lambda r: resize_norm(r, 64, (0, 0, 0), (1, 1, 1))
            preps['224_no_norm'] = lambda r: resize_norm(r, 224, (0, 0, 0), (1, 1, 1))
            preps['64_imagenet_norm'] = lambda r: resize_norm(r, 64, mean, std)
            return net, src, m, preps, None
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=m))
        preps['native'] = lambda r: torch.stack([tf(im) for im in pil(r)])
        return net, src, m, preps, 'native'
    if kind == 'hf':
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        m = AutoModelForImageClassification.from_pretrained(repo).eval()
        src = [m.config.id2label[i] for i in range(len(m.config.id2label))]
        size, mean, std = hf_prep_cfg(repo)
        net = lambda x: F.softmax(m(pixel_values=x).logits, 1)
        preps = {'plain': lambda r: resize_norm(r, size, mean, std)}
        try:
            proc = AutoImageProcessor.from_pretrained(repo); proc(images=pil(RAW[:2]), return_tensors='pt')
            preps['native'] = lambda r: proc(images=pil(r), return_tensors='pt')['pixel_values']
        except Exception as e:
            print('  processor unusable:', type(e).__name__, str(e)[:80])
        return net, src, m, preps, 'native' if 'native' in preps else 'plain'
    if kind in ('clipft', 'clipfull'):
        from transformers import CLIPModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
        base = extra[0] if kind == 'clipft' else repo
        cm = CLIPModel.from_pretrained(base).eval(); tok = CLIPTokenizer.from_pretrained(base)
        if kind == 'clipft':  # fine-tuned vision tower; projection + text head from the original CLIP (FusionBench convention)
            vm = CLIPVisionModel.from_pretrained(repo); print('  load', cm.vision_model.load_state_dict(vm.state_dict()))
        src = extra[1] if kind == 'clipft' else extra
        with torch.inference_mode():
            W = []
            for c in src:
                t = tok([tp.format(c) for tp in TEMPLATES], padding=True, return_tensors='pt')
                e = F.normalize(_emb(cm.get_text_features(**t)), dim=-1).mean(0); W.append(F.normalize(e, dim=0))
            W = torch.stack(W)
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        scale = cm.logit_scale.exp()
        class V(torch.nn.Module):
            def __init__(s): super().__init__(); s.cm = cm
            def forward(s, x): return F.normalize(_emb(s.cm.get_image_features(pixel_values=x)), dim=-1) @ W.T * scale
        v = V().eval(); proc = CLIPImageProcessor.from_pretrained(base)
        net = lambda x: F.softmax(v(x), 1)
        preps = {'plain': lambda r: resize_norm(r, 224, mean, std),
                 'native': lambda r: proc(images=pil(r), return_tensors='pt')['pixel_values']}
        return net, src, v, preps, 'native'
    raise ValueError(kind)

def mapping(src):
    M = np.zeros((len(src), len(CLASSES)), np.float32); mapped = []
    for i, s in enumerate(src):
        t = MAP.get(norm(s))
        for c in ((t,) if isinstance(t, str) else (t or ())): M[i, CLASSES.index(c)] = 1
        if t: mapped.append(i)
    return M, mapped

def run(key):
    out = f'{OUT}/{key}.json'
    if os.path.exists(out): return
    t0 = time.time(); print('==', key, flush=True); torch.set_num_threads(32)
    net, src, model, preps, primary = build(key)
    M, mapped = mapping(src)
    res = dict(key=key, repo=ZOO[key][1], trained_on=ZOO[key][3], n_src_labels=len(src),
               unmapped_labels=[s for i, s in enumerate(src) if i not in mapped], covers=[c for j, c in enumerate(CLASSES) if M[:, j].any()],
               params_m=sum(p.numel() for p in model.parameters()) / 1e6, by_prep={})
    best = None
    for pname, prep in preps.items():
        with torch.inference_mode():
            P = torch.cat([net(prep(RAW[i:i + 64])) for i in range(0, len(RAW), 64)]).numpy()
        pr = (P @ M).argmax(1); ok = pr == y
        res['by_prep'][pname] = dict(acc_eval=float(ok[is_eval].mean()), acc_all1260=float(ok.mean()), input=int(prep(RAW[:1]).shape[-1]))
        if (primary is None and (best is None or ok.mean() > best[1].mean())) or pname == primary: best = (pname, ok, P)
    pname, ok, P = best; top1 = P.argmax(1)
    res.update(primary_prep=pname if primary else pname + ' (undocumented; best of variants tried)',
               input=res['by_prep'][pname]['input'], acc_eval=float(ok[is_eval].mean()), acc_all1260=float(ok.mean()),
               in_vocab_rate=float(np.isin(top1, mapped).mean()), unmapped_mass=float(1 - P[:, mapped].sum(1).mean()),
               per_class_recall_all={c: float(ok[y == j].mean()) for j, c in enumerate(CLASSES)})
    try: leak = json.load(open(f'{S}/leakcheck.json'))['assign']
    except Exception: leak = {}
    ds = {'cm93': 'cm93/eurosat', 'tanganke': 'tanganke/eurosat'}
    for pre, d in ds.items():  # score separately on our tiles that sit (exact match) in that dataset's train vs held-out splits
        if key.startswith(pre) and d in leak:
            where = np.array([leak[d].get(n, '') for n in names])
            for split in ('train', 'validation', 'test'):
                idx = where == split
                if idx.any(): res[f'acc_on_our_tiles_in_{split}_split'] = (float(ok[idx].mean()), int(idx.sum()))
    res['wall_s'] = time.time() - t0
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ['key', 'primary_prep', 'acc_eval', 'acc_all1260', 'in_vocab_rate', 'params_m']}),
          {k: round(v['acc_all1260'], 3) for k, v in res['by_prep'].items()}, {k: v for k, v in res.items() if k.startswith('acc_on')}, flush=True)

def latency(key):
    net, src, model, preps, primary = build(key); j = json.load(open(f'{OUT}/{key}.json'))
    prep = preps[j['primary_prep'].split(' ')[0]]; r1 = RAW[:1]; res = {}
    x1 = prep(r1); fwd = lambda r: net(x1)  # model-only time (preprocessing excluded), comparable with the backbone table
    for th in (1, 4):
        torch.set_num_threads(th)
        with torch.inference_mode():
            for _ in range(3): fwd(r1)
            ts = []
            for _ in range(15):
                t = time.perf_counter(); fwd(r1); ts.append((time.perf_counter() - t) * 1000)
        res[f'torch_ms_{th}thr'] = float(np.median(ts))
    j.update(res); json.dump(j, open(f'{OUT}/{key}.json', 'w'), indent=1)
    print(key, res, flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]; lat = '--latency' in args; force = '--force' in args  # --force: recompute even if spec/<key>.json exists
    args = [a for a in args if a not in ('--latency', '--force')] or list(ZOO)
    for k in args:
        if force and not lat and os.path.exists(f'{OUT}/{k}.json'): os.remove(f'{OUT}/{k}.json')
        try: latency(k) if lat else run(k)
        except Exception as e:
            import traceback; traceback.print_exc(); print('FAILED', k, type(e).__name__, e, flush=True)

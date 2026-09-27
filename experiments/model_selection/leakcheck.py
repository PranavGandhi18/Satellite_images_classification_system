"""Assign each of our 1,260 tiles to exactly one split of each public EuroSAT copy.
Exact pixel-hash match first; tiles with no exact match anywhere get the single nearest image across all splits
(reported separately, with its RMS distance)."""
import glob, os, io, json, hashlib, collections, numpy as np, pandas as pd
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
R = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'be-mlsys-assignment-dataset'))
S = os.path.dirname(os.path.abspath(__file__))
labels = dict(l.strip().split(',') for l in open(f'{R}/eval_labels.csv').readlines()[1:])
ours = sorted(glob.glob(f'{R}/candidate_tiles/*/*.png')) + sorted(glob.glob(f'{R}/eval_set/*.png'))
name = lambda p: ('eval/' if 'eval_set' in p else 'cand/') + os.path.basename(p)
A = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in ours]); Af = A.reshape(len(A), -1).astype(np.float32)
Hs = [hashlib.md5(a.tobytes()).hexdigest() for a in A]
out, summary = {}, {}
for rid in ['cm93/eurosat', 'nielsr/eurosat-demo', 'tanganke/eurosat']:
    files = [s.rfilename for s in HfApi().dataset_info(rid).siblings if s.rfilename.endswith('.parquet') and any(k in s.rfilename for k in ('train', 'test', 'validation'))]
    where = collections.defaultdict(set); allimg, allsplit = [], []
    for f in files:
        split = [k for k in ('train', 'test', 'validation') if k in os.path.basename(f)][0]
        df = pd.read_parquet(hf_hub_download(rid, f, repo_type='dataset')); col = [c for c in df.columns if 'image' in c.lower() or 'img' in c.lower()][0]
        imgs = np.stack([np.asarray(Image.open(io.BytesIO(d['bytes'])).convert('RGB').resize((64, 64))) for d in df[col]])
        for h in (hashlib.md5(x.tobytes()).hexdigest() for x in imgs): where[h].add(split)
        allimg.append(imgs.reshape(len(imgs), -1)); allsplit += [split] * len(imgs)
    B = np.concatenate(allimg).astype(np.float32); Bn = (B ** 2).sum(1); allsplit = np.array(allsplit)
    assign, stats = {}, collections.Counter()
    for i, p in enumerate(ours):
        s = where.get(Hs[i])
        if s:
            assign[name(p)] = sorted(s)[0] if len(s) == 1 else 'multiple:' + '+'.join(sorted(s)); stats['exact' if len(s) == 1 else 'exact_multi'] += 1
        else:
            d = np.sqrt(np.maximum((Af[i] ** 2).sum() - 2 * B @ Af[i] + Bn, 0) / Af.shape[1]); j = int(d.argmin())
            assign[name(p)] = f'near:{allsplit[j]}:{d[j]:.2f}'; stats['near'] += 1
    splits = collections.Counter(v if not v.startswith('near') else 'near:' + v.split(':')[1] for v in assign.values())
    ev = collections.Counter(v if not v.startswith('near') else 'near:' + v.split(':')[1] for k, v in assign.items() if k.startswith('eval'))
    out[rid] = assign; summary[rid] = dict(match=dict(stats), all_1260=dict(splits), eval_210=dict(ev))
    print(rid, json.dumps(summary[rid]), flush=True)
json.dump(dict(assign=out, summary=summary), open(f'{S}/leakcheck.json', 'w'))

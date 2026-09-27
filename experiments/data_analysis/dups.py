import glob, hashlib, os, collections
import numpy as np, imagehash
from PIL import Image
R = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'be-mlsys-assignment-dataset'))
labels = dict(l.strip().split(',') for l in open(R+'/eval_labels.csv').readlines()[1:])
cand = sorted(glob.glob(R+'/candidate_tiles/*/*.png')); ev = sorted(glob.glob(R+'/eval_set/*.png'))
def pixhash(p): return hashlib.md5(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest()
H = collections.defaultdict(list)
for p in cand+ev: H[pixhash(p)].append(os.path.relpath(p,R))
d = [v for v in H.values() if len(v)>1]
print('exact pixel duplicate groups:', len(d)); [print(' ', g) for g in d[:20]]
# near-duplicates via phash, also rotated/flipped versions
def hashes(p):
    im = Image.open(p).convert('RGB')
    return [imagehash.phash(t) for t in [im, im.transpose(Image.FLIP_LEFT_RIGHT), im.transpose(Image.FLIP_TOP_BOTTOM), im.rotate(90), im.rotate(180), im.rotate(270)]]
ch = {p: imagehash.phash(Image.open(p).convert('RGB')) for p in cand}
near=[]
for e in ev:
    hs = hashes(e)
    best = min(((min(h - ch[c] for h in hs), c) for c in ch))
    if best[0] <= 6: near.append((os.path.basename(e), labels[os.path.basename(e)], best[0], os.path.relpath(best[1],R)))
print('eval tiles with near-dup (phash<=6, incl. flips/rot) in candidates:', len(near)); [print(' ',n) for n in near[:20]]
# pixel-level nearest neighbour L2 distance eval -> cand
C = np.stack([np.asarray(Image.open(p).convert('RGB'),dtype=np.float32).ravel() for p in cand])
E = np.stack([np.asarray(Image.open(p).convert('RGB'),dtype=np.float32).ravel() for p in ev])
d2 = ((E**2).sum(1)[:,None] - 2*E@C.T + (C**2).sum(1)[None]).clip(0)
nn = np.sqrt(d2.min(1)/C.shape[1])
print('eval->cand nearest RMS pixel dist: min %.2f median %.2f' % (nn.min(), np.median(nn)))
# 1-NN raw-pixel accuracy as crude baseline
cl = [p.split('/')[-2] for p in cand]
pred = [cl[i] for i in d2.argmin(1)]
acc = np.mean([pred[i]==labels[os.path.basename(e)] for i,e in enumerate(ev)])
print('1-NN raw pixel accuracy on eval: %.3f' % acc)

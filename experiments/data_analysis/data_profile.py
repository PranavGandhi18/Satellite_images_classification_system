import os, glob, collections, hashlib
import numpy as np
from PIL import Image
R = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'be-mlsys-assignment-dataset'))
files = sorted(glob.glob(R+'/candidate_tiles/*/*.png')) + sorted(glob.glob(R+'/eval_set/*.png'))
props = collections.Counter()
for f in files:
    im = Image.open(f)
    props[(im.size, im.mode, im.format, tuple(sorted(im.info.keys())))] += 1
print('image properties:', props)
# check any 16-bit / alpha
im = Image.open(files[0]); print('sample info', im.info, im.getbands())
# per-class stats
labels = dict(l.strip().split(',') for l in open(R+'/eval_labels.csv').readlines()[1:])
def stats(paths):
    a = np.stack([np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) for p in paths])
    mean = a.mean(axis=(0,1,2)); std = a.reshape(-1,3).std(0)
    per_img_std = a.reshape(len(a),-1,3).std(1).mean(0)
    return mean, std, per_img_std, a
print('\nclass            split   mean RGB              global std          within-img std')
allc=[]
for c in sorted(os.listdir(R+'/candidate_tiles')):
    cp = sorted(glob.glob(f'{R}/candidate_tiles/{c}/*.png'))
    ep = [f'{R}/eval_set/{k}' for k,v in labels.items() if v==c]
    for name, ps in [('cand',cp),('eval',ep)]:
        m,s,w,_ = stats(ps)
        print(f'{c:14s} {name}  {np.round(m,1)}  {np.round(s,1)}  {np.round(w,1)}')
# file sizes
sz = [os.path.getsize(f) for f in files]; print('\nfile size bytes min/median/max', min(sz), int(np.median(sz)), max(sz))
# pixel value range / saturation
a = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in files])
print('pixel min/max', a.min(), a.max(), 'frac saturated 255:', (a==255).mean(), 'frac 0:', (a==0).mean())
# near-black / blank / constant tiles
per = a.reshape(len(a),-1,3)
lo = [(files[i], per[i].mean()) for i in range(len(a)) if per[i].std()<3]
print('near-constant tiles:', lo[:10])
dark = sorted([(per[i].mean(), os.path.relpath(files[i],R)) for i in range(len(a))])[:5]
bright = sorted([(per[i].mean(), os.path.relpath(files[i],R)) for i in range(len(a))])[-5:]
print('darkest', dark); print('brightest', bright)

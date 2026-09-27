"""Full fine-tune of a pretrained backbone on candidate_tiles, CPU only.

Fixed recipe chosen up-front (no tuning on eval): AdamW, cosine LR, 15 epochs, label smoothing 0.1,
flips + 90-degree rotations (overhead imagery has no canonical orientation).
15% of candidates are held out as a validation split for picking the best epoch; eval_set is scored once.
usage: python finetune.py <timm_name> <input_size> <backbone_lr> [threads]
"""
import os, sys, glob, json, time, math, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F, timm
from PIL import Image
from sklearn.model_selection import train_test_split

R = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'be-mlsys-assignment-dataset'))
S = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']
name, size, blr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 32); torch.manual_seed(0); np.random.seed(0)
EPOCHS, BS = 15, 32

labels = dict(l.strip().split(',') for l in open(f'{R}/eval_labels.csv').readlines()[1:])
cand = sorted(glob.glob(f'{R}/candidate_tiles/*/*.png')); ev = sorted(glob.glob(f'{R}/eval_set/*.png'))
load = lambda ps: torch.from_numpy(np.stack([np.asarray(Image.open(p).convert('RGB')) for p in ps])).permute(0, 3, 1, 2).contiguous()
Xc, Xe = load(cand), load(ev)
yc = torch.tensor([CLASSES.index(p.split('/')[-2]) for p in cand]); ye = torch.tensor([CLASSES.index(labels[os.path.basename(p)]) for p in ev])
tr, va = train_test_split(np.arange(len(cand)), test_size=0.15, stratify=yc.numpy(), random_state=0)

if name == 'ssl4eo_resnet18_rgb_moco':  # torchgeo SSL4EO-S12 weights; expects reflectance (DN/1e4). 255 <-> 0.3, picked by CV in the probe
    import torchgeo.models as TG
    m = TG.resnet18(weights=TG.ResNet18_Weights.SENTINEL2_RGB_MOCO); m.reset_classifier(len(CLASSES))
    mean = torch.zeros(1, 3, 1, 1); std = torch.full((1, 3, 1, 1), 1 / 0.3)
else:
    kw = dict(dynamic_img_size=True) if 'vit' in name and size != 224 else {}
    m = timm.create_model(name, pretrained=True, num_classes=len(CLASSES), **kw)
    cfg = m.pretrained_cfg; mean = torch.tensor(cfg['mean']).view(1, 3, 1, 1); std = torch.tensor(cfg['std']).view(1, 3, 1, 1)

def prep(x, aug=False):
    x = x.float() / 255.
    if aug:
        if np.random.rand() < .5: x = x.flip(3)
        if np.random.rand() < .5: x = x.flip(2)
        x = torch.rot90(x, int(np.random.randint(4)), (2, 3))
    if size != x.shape[-1]: x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False)
    return (x - mean) / std

head = list(m.get_classifier().parameters())
hid = {id(p) for p in head}
opt = torch.optim.AdamW([{'params': [p for p in m.parameters() if id(p) not in hid], 'lr': blr},
                         {'params': head, 'lr': blr * 10}], weight_decay=0.05)
steps = EPOCHS * math.ceil(len(tr) / BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[blr, blr * 10], total_steps=steps, pct_start=0.1)

def predict(X):
    m.eval(); out = []
    with torch.inference_mode():
        for i in range(0, len(X), 128): out.append(F.softmax(m(prep(X[i:i + 128])), 1))
    return torch.cat(out)

best = (-1, None); t0 = time.time()
for ep in range(EPOCHS):
    m.train(); perm = np.random.permutation(tr)
    for i in range(0, len(perm), BS):
        b = perm[i:i + BS]
        loss = F.cross_entropy(m(prep(Xc[b], aug=True)), yc[b], label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    va_acc = (predict(Xc[va]).argmax(1) == yc[va]).float().mean().item()
    print(f'{name}@{size} ep{ep} loss {loss.item():.3f} val {va_acc:.3f} {time.time() - t0:.0f}s', flush=True)
    if va_acc >= best[0]: best = (va_acc, {k: v.clone() for k, v in m.state_dict().items()}, ep)
m.load_state_dict(best[1])
P = predict(Xe); pr = P.argmax(1); acc = (pr == ye).float().mean().item()
conf = P.max(1).values; keep = torch.argsort(-conf)[:int(0.8 * len(ye))]
res = dict(model=name, input=size, backbone_lr=blr, epochs=EPOCHS, best_epoch=best[2], val_acc=best[0], eval_acc=acc,
           acc_at_80cov=(pr[keep] == ye[keep]).float().mean().item(), train_minutes_cpu=(time.time() - t0) / 60,
           threads=torch.get_num_threads(),
           per_class_recall={c: (pr[ye == i] == i).float().mean().item() for i, c in enumerate(CLASSES)})
print(json.dumps(res)); os.makedirs(f'{S}/ft', exist_ok=True)
json.dump(res, open(f'{S}/ft/{name}@{size}.json', 'w'), indent=1)

"""Fine-tune the land-use classifier on CPU and write a versioned, self-describing model artifact.

The recipe is the one benchmarked in model_selection_thoughtprocess.md (DINOv3 ViT-S/16 @ 112 px; 15 epochs, AdamW,
one-cycle LR, label smoothing 0.1, flips + 90-degree rotations). eval_set is never touched here: the epoch, the
temperature, the confidence threshold and the unfamiliarity threshold are all chosen on a validation split carved
out of candidate_tiles.

Writes models/<version>/:
  model.onnx                raw tile (float32 0-255, N x 3 x 64 x 64) -> logits[N, 7], embedding[N, D];
                            preprocessing (/255, bicubic 64->112, normalise) is *inside* the graph
  manifest.json             classes in index order, preprocessing, checksums, training-data hash, calibration,
                            decision policy, self-test expectations, metrics
  reference_embeddings.npy  L2-normalised embeddings of the training tiles (for the "unfamiliar" score)
  selftest/<class>.png      one golden tile per class, re-classified by the service at startup
and points models/active at it (unless --no-activate).

Runs fully offline: the pretrained backbone is read from weights/<timm_name>/model.safetensors (fetched once by
scripts/fetch_offline_assets.py) and the Hugging Face libraries are forced into offline mode before they are imported.

usage: python training/train.py [--backbone dinov3_vits16|vit_s16_in21k] [--epochs 15] [--threads 32]
"""
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')        # must be set before timm / huggingface_hub are imported
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.backbones import BACKBONES, weights_path  # noqa: E402

DATASET = Path(os.environ.get('DATASET_DIR', ROOT / 'be-mlsys-assignment-dataset'))
CLASSES = ['AnnualCrop', 'Forest', 'Highway', 'Industrial', 'Residential', 'River', 'SeaLake']
TILE = 64


class Preprocess(nn.Module):
    """Raw tile (float 0-255, N x 3 x 64 x 64) -> normalised N x 3 x S x S. Exported inside the ONNX graph."""

    def __init__(self, size, mean, std):
        super().__init__()
        self.size = size
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x / 255.0, size=(self.size, self.size), mode='bicubic', align_corners=False)
        return (x - self.mean) / self.std


class TileClassifier(nn.Module):
    """Preprocess + backbone. Returns (logits, embedding); the embedding is the pre-logits feature vector."""

    def __init__(self, backbone, prep):
        super().__init__()
        self.prep, self.backbone = prep, backbone

    def forward(self, x):
        emb = self.backbone.forward_head(self.backbone.forward_features(self.prep(x)), pre_logits=True)
        return self.backbone.get_classifier()(emb), emb


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_candidates():
    classes = sorted(p.name for p in (DATASET / 'candidate_tiles').iterdir() if p.is_dir())
    assert classes == CLASSES, f'unexpected class folders: {classes}'
    paths = sorted((DATASET / 'candidate_tiles').glob('*/*.png'))
    x = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in paths])  # N, 64, 64, 3 uint8
    y = np.array([CLASSES.index(p.parent.name) for p in paths])
    data_hash = hashlib.sha256('\n'.join(f'{p.relative_to(DATASET)}:{sha256_file(p)}' for p in paths).encode()).hexdigest()
    return paths, torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(y), data_hash


def augment(x):
    """Flips and 90-degree rotations: overhead imagery has no canonical orientation."""
    if random.random() < 0.5: x = x.flip(3)
    if random.random() < 0.5: x = x.flip(2)
    return torch.rot90(x, random.randrange(4), (2, 3))


@torch.inference_mode()
def run(model, x, bs=128):
    model.eval()
    outs = [model(x[i:i + bs].float()) for i in range(0, len(x), bs)]
    return torch.cat([o[0] for o in outs]).numpy(), torch.cat([o[1] for o in outs]).numpy()


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, y, t):
    return float(-np.log(softmax(logits / t)[np.arange(len(y)), y] + 1e-12).mean())


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def choose_min_confidence(conf, correct, target, floor):
    """Smallest threshold >= floor whose accepted set reaches `target` accuracy on validation (= max coverage)."""
    for t in sorted({floor, *conf[conf >= floor].tolist()}):
        keep = conf >= t
        if keep.any() and correct[keep].mean() >= target:
            return float(t), float(keep.mean()), float(correct[keep].mean())
    return 1.0, 0.0, float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backbone', choices=BACKBONES, default='dinov3_vits16')
    ap.add_argument('--size', type=int, default=112, help='input size the 64 px tiles are upsampled to')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--threads', type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument('--target-accuracy', type=float, default=0.99, help='validation accuracy required of "confident" predictions')
    ap.add_argument('--min-confidence-floor', type=float, default=0.5)
    ap.add_argument('--ood-percentile', type=float, default=99.0, help='validation percentile that sets the unfamiliarity threshold')
    ap.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    ap.add_argument('--no-activate', action='store_true', help='do not repoint models/active at the new artifact')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    timm_name, backbone_lr, licence = BACKBONES[args.backbone]
    pretrained_file = weights_path(timm_name)
    if not pretrained_file.exists():
        raise SystemExit(f'pretrained weights not found at {pretrained_file}\n'
                         f'run `python scripts/fetch_offline_assets.py` once on a machine with internet access')
    t_start = time.time()

    paths, X, Y, data_hash = load_candidates()
    tr, va = train_test_split(np.arange(len(Y)), test_size=0.15, stratify=Y.numpy(), random_state=0)
    print(f'{len(tr)} train / {len(va)} validation tiles; backbone {timm_name} @ {args.size}px; {args.threads} threads', flush=True)

    # local file only (custom_load=False: use timm's normal safetensors loader, which also resizes position embeddings)
    backbone = timm.create_model(timm_name, pretrained=True, num_classes=len(CLASSES), img_size=args.size,
                                 pretrained_cfg_overlay=dict(file=str(pretrained_file), custom_load=False))
    cfg = backbone.pretrained_cfg
    mean, std = list(cfg['mean']), list(cfg['std'])
    model = TileClassifier(backbone, Preprocess(args.size, mean, std))

    head = list(backbone.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    opt = torch.optim.AdamW([{'params': [p for p in backbone.parameters() if id(p) not in head_ids], 'lr': backbone_lr},
                             {'params': head, 'lr': backbone_lr * 10}], weight_decay=0.05)
    steps = args.epochs * -(-len(tr) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[backbone_lr, backbone_lr * 10], total_steps=steps, pct_start=0.1)

    best, log = (-1.0, None, -1), []
    for ep in range(args.epochs):
        model.train()
        perm, losses = np.random.permutation(tr), []
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            logits, _ = model(augment(X[b].float()))
            loss = F.cross_entropy(logits, Y[b], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        val_acc = float((run(model, X[va])[0].argmax(1) == Y[va].numpy()).mean())
        log.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_acc=val_acc, elapsed_s=round(time.time() - t_start)))
        print(f'epoch {ep:2d}  loss {np.mean(losses):.3f}  val acc {val_acc:.3f}  {time.time() - t_start:.0f}s', flush=True)
        if val_acc >= best[0]:
            best = (val_acc, {k: v.detach().clone() for k, v in backbone.state_dict().items()}, ep)
    backbone.load_state_dict(best[1])
    train_minutes = (time.time() - t_start) / 60

    # --- calibration, decision policy, unfamiliarity reference: all from train/validation only ---
    lg_tr, emb_tr = run(model, X[tr])
    lg_va, emb_va = run(model, X[va])
    y_va = Y[va].numpy()
    temps = np.exp(np.linspace(np.log(0.05), np.log(10), 400))
    temperature = float(temps[np.argmin([nll(lg_va, y_va, t) for t in temps])])
    p_va = softmax(lg_va / temperature)
    conf_va, correct_va = p_va.max(1), p_va.argmax(1) == y_va
    min_conf, coverage, sel_acc = choose_min_confidence(conf_va, correct_va, args.target_accuracy, args.min_confidence_floor)

    refs = emb_tr / np.linalg.norm(emb_tr, axis=1, keepdims=True)
    d_va = 1.0 - ((emb_va / np.linalg.norm(emb_va, axis=1, keepdims=True)) @ refs.T).max(1)
    ood_threshold = float(np.percentile(d_va, args.ood_percentile))

    # --- export: same weights, non-fused attention + fixed input size so the TorchScript exporter can trace it ---
    version = f"{args.backbone}-{args.size}px-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}"
    out = args.models_dir / version
    (out / 'selftest').mkdir(parents=True)
    timm.layers.set_fused_attn(False)
    export_bb = timm.create_model(timm_name, pretrained=False, num_classes=len(CLASSES), img_size=args.size)
    export_bb.load_state_dict(backbone.state_dict())
    export_model = TileClassifier(export_bb, Preprocess(args.size, mean, std)).eval()
    torch.onnx.export(export_model, torch.zeros(1, 3, TILE, TILE), out / 'model.onnx', input_names=['tile'],
                      output_names=['logits', 'embedding'], opset_version=18, dynamo=False,
                      dynamic_axes={'tile': {0: 'n'}, 'logits': {0: 'n'}, 'embedding': {0: 'n'}})
    sess = ort.InferenceSession(str(out / 'model.onnx'), providers=['CPUExecutionProvider'])
    onnx_lg = np.concatenate([sess.run(['logits'], {'tile': X[va][i:i + 64].float().numpy()})[0] for i in range(0, len(va), 64)])
    parity = dict(max_abs_logit_diff=float(np.abs(onnx_lg - lg_va).max()), argmax_agreement=float((onnx_lg.argmax(1) == lg_va.argmax(1)).mean()))
    assert parity['argmax_agreement'] == 1.0, f'ONNX export disagrees with PyTorch: {parity}'

    np.save(out / 'reference_embeddings.npy', refs.astype(np.float32))

    # one golden tile per class: the most confident *correct* validation tile; expectations come from the ONNX run
    selftest = []
    for c, name in enumerate(CLASSES):
        idx = [i for i in range(len(va)) if y_va[i] == c and correct_va[i]]
        j = max(idx, key=lambda i: conf_va[i])
        dst = out / 'selftest' / f'{name}.png'
        shutil.copyfile(paths[va[j]], dst)
        lg = sess.run(['logits'], {'tile': X[va[j]][None].float().numpy()})[0]
        selftest.append(dict(file=f'selftest/{name}.png', source=str(paths[va[j]].relative_to(DATASET)), expected_label=name,
                             expected_probabilities=softmax(lg / temperature)[0].round(6).tolist()))

    manifest = dict(
        model_version=version,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        task='single-label land-use classification of 64x64 Sentinel-2 RGB tiles',
        classes=CLASSES,
        input=dict(format='PNG', width=TILE, height=TILE, bands=3, mode='RGB', dtype='uint8',
                   tensor='float32 N x 3 x 64 x 64, raw 0-255 values, channel order R,G,B'),
        preprocessing=dict(inside_onnx=True, steps=['x / 255', f'bicubic resize {TILE}->{args.size} (align_corners=False)', '(x - mean) / std'],
                           size=args.size, mean=mean, std=std),
        onnx=dict(file='model.onnx', sha256=sha256_file(out / 'model.onnx'), input='tile', outputs=['logits', 'embedding'],
                  embedding_dim=int(emb_tr.shape[1]), opset=18, parity_vs_pytorch_on_validation=parity),
        backbone=dict(key=args.backbone, timm_name=timm_name, weights_licence=licence,
                      pretrained_weights=dict(file=str(pretrained_file.relative_to(ROOT)), sha256=sha256_file(pretrained_file)),
                      params_m=round(sum(p.numel() for p in backbone.parameters()) / 1e6, 2)),
        training=dict(data='candidate_tiles', data_sha256=data_hash, n_train=len(tr), n_val=len(va), split='stratified 85/15, seed 0',
                      recipe=dict(epochs=args.epochs, batch_size=args.batch_size, optimizer='AdamW', weight_decay=0.05, backbone_lr=backbone_lr,
                                  head_lr=backbone_lr * 10, schedule='one-cycle (10% warm-up)', label_smoothing=0.1, augmentation='h/v flips, 90-degree rotations'),
                      best_epoch=best[2], val_accuracy=best[0], train_minutes_cpu=round(train_minutes, 1), threads=args.threads, log=log,
                      versions=dict(torch=torch.__version__, timm=timm.__version__, onnxruntime=ort.__version__)),
        calibration=dict(method='temperature scaling on validation (grid search, NLL)', temperature=round(temperature, 4),
                         val_nll_before=nll(lg_va, y_va, 1.0), val_nll_after=nll(lg_va, y_va, temperature),
                         val_ece_before=ece(softmax(lg_va), y_va), val_ece_after=ece(p_va, y_va)),
        policy=dict(version='p1', min_confidence=round(min_conf, 6), ood_threshold=round(ood_threshold, 6),
                    rules=['unfamiliar if ood_score > ood_threshold',
                           'needs_review if confidence < min_confidence or quality flags include blank / grey_low_texture',
                           'confident otherwise'],
                    chosen_on='validation split', target_selective_accuracy=args.target_accuracy,
                    val_coverage_at_min_confidence=coverage, val_accuracy_at_min_confidence=sel_acc,
                    ood_percentile=args.ood_percentile),
        ood=dict(method='1 - max cosine similarity between the tile embedding and the training-tile embeddings',
                 reference_file='reference_embeddings.npy', reference_sha256=sha256_file(out / 'reference_embeddings.npy'),
                 n_reference=len(tr)),
        selftest=dict(tolerance=1e-3, tiles=selftest),
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    if not args.no_activate:
        link, tmp = args.models_dir / 'active', args.models_dir / '.active.tmp'
        if tmp.is_symlink() or tmp.exists(): tmp.unlink()
        tmp.symlink_to(version)
        os.replace(tmp, link)
    print(json.dumps(dict(version=version, val_accuracy=best[0], best_epoch=best[2], temperature=temperature, min_confidence=min_conf,
                          val_coverage=coverage, ood_threshold=ood_threshold, parity=parity, train_minutes=round(train_minutes, 1),
                          activated=not args.no_activate), indent=1))


if __name__ == '__main__':
    main()

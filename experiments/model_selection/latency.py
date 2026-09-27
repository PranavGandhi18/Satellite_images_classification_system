"""Quiet-machine CPU cost pass: PyTorch eager vs ONNX Runtime, batch 1, 1 and 4 threads, plus batch-32 throughput.
Run only when nothing else is using the CPU. Results -> latency/<key>.json
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
sys.argv, _argv = sys.argv[:1], sys.argv[1:]
import probe_bench as bench
from probe_bench import ZOO, FE, tg_resnet, tg_swin_t_satlas, openclip, CLIPVis
S = bench.S; OUT = f'{S}/latency'; os.makedirs(OUT, exist_ok=True); ONNXD = f'{S}/onnx'; os.makedirs(ONNXD, exist_ok=True)

def make(key):
    group, fam, b, size = ZOO[key]
    size = json.load(open(f'{FE}/{key}.json'))['input']
    if isinstance(b, str):
        _, arch, *w = b.split(':'); m = tg_swin_t_satlas() if arch == 'swin_t' else tg_resnet(arch, w[0])
    elif isinstance(b, tuple):
        m = CLIPVis(openclip(b[1], b[2], b[3])[0])
    else:
        m = b()[0]
    return m.eval(), size

def timeit(fn, n):
    for _ in range(3): fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))

def run(key):
    import onnxruntime as ort
    m, size = make(key); x1 = torch.randn(1, 3, size, size); x32 = torch.randn(32, 3, size, size)
    big = sum(p.numel() for p in m.parameters()) > 1e8; n = 5 if big else 30
    res = dict(key=key, input=size)
    for th in (1, 4):
        torch.set_num_threads(th)
        with torch.inference_mode(): res[f'torch_ms_{th}thr'] = timeit(lambda: m(x1), n)
    with torch.inference_mode(): res['torch_tiles_per_s_4thr_b32'] = 32 / (timeit(lambda: m(x32), 3 if big else 8) / 1000)
    path = f'{ONNXD}/{key}.onnx'
    try:
        torch.set_num_threads(8)
        torch.onnx.export(m, x1, path, input_names=['x'], output_names=['f'], dynamic_axes={'x': {0: 'n'}}, opset_version=18, dynamo=False)
        res['onnx_mb'] = (os.path.getsize(path) + sum(os.path.getsize(f'{ONNXD}/{f}') for f in os.listdir(ONNXD) if f.startswith(key + '.onnx') and f != key + '.onnx')) / 1e6
        with torch.inference_mode(): ref = m(x1).numpy()
        for th in (1, 4):
            so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
            xn = x1.numpy()
            if th == 1: res['onnx_max_abs_diff'] = float(np.abs(sess.run(None, {'x': xn})[0].reshape(ref.shape) - ref).max())
            res[f'onnx_ms_{th}thr'] = timeit(lambda: sess.run(None, {'x': xn}), n)
            if th == 4: res['onnx_tiles_per_s_4thr_b32'] = 32 / (timeit(lambda: sess.run(None, {'x': x32.numpy()}), 3 if big else 8) / 1000)
    except Exception as e:
        res['onnx_error'] = str(e)[:200]
    for f in os.listdir(ONNXD):
        if f.startswith(key): os.remove(f'{ONNXD}/{f}')
    json.dump(res, open(f'{OUT}/{key}.json', 'w'), indent=1); print(json.dumps(res), flush=True)

if __name__ == '__main__':
    for k in (_argv or [k for k in ZOO if os.path.exists(f'{FE}/{k}.json')]):
        try: run(k)
        except Exception as e: print('FAILED', k, e, flush=True)

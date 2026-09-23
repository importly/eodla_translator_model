"""Measure the irreducible noise floor per (target size, normalisation).

Two independent Poisson realisations of the same (input, kernel) differ by
sqrt(2) x noise, so MSE(a, b) / 2 is the per-pixel noise variance - the best MSE
any model could achieve. Also reports the conv baseline so the achievable range
is bracketed at both ends.

Re-run this whenever the dataset is regenerated and paste the numbers into
train.FLOOR, or every "x floor" figure in the repo is wrong:

    uv run python reference/analysis/floor.py

Pairs come from gen_test.h5, so the floor is measured on exactly the held-out
rows the models are scored on. The stored target48 is a third realisation of the
same pairs, which gives a free check on the simulator: corr(mine, stored) should
match corr(mine_a, mine_b).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from eodla_sim import TorchEODLASim
from data import NORMS, pool_to, make_conv
from paths import DATA

K = 96
CROP = 144
device = "cuda"

d = np.load(DATA / "operands.npz")
inputs = torch.from_numpy(d["inputs"]).to(device)
kernels = torch.from_numpy(d["kernels"]).to(device)

with h5py.File(DATA / "gen_test.h5", "r", locking=False) as f:
    ii = torch.from_numpy(f["ii"][:K].astype(np.int64)).to(device)
    kk = torch.from_numpy(f["kk"][:K].astype(np.int64)).to(device)
    orig48 = torch.from_numpy(f["target48"][:K]).to(device)

s = TorchEODLASim()
B = 8


def sim48(seed):
    g = torch.Generator(device=device); g.manual_seed(seed)
    outs = []
    for p in range(0, K, B):
        full = s.simulate(inputs[ii[p:p + B]], kernels[kk[p:p + B]], gen=g)
        H, W = full.shape[-2:]
        r0, c0 = (H - CROP) // 2, (W - CROP) // 2
        q = full[..., r0:r0 + CROP, c0:c0 + CROP].unsqueeze(1)
        outs.append(F.interpolate(q, size=(48, 48), mode="area").squeeze(1))
    return torch.cat(outs)


print(f"simulating {K} held-out pairs twice ...")
a48, b48 = sim48(1), sim48(2)
o48 = orig48


def corr(a, b):
    a = (a - a.mean((-2, -1), keepdim=True)).flatten(1)
    b = (b - b.mean((-2, -1), keepdim=True)).flatten(1)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)).mean().item()


print(f"\nsimulator fidelity on held-out (48x48 raw): "
      f"corr(mine,stored) {corr(a48, o48):+.4f}   corr(a,b) {corr(a48, b48):+.4f}")

print(f"\n{'size':>5s} {'norm':>8s} {'floor MSE':>12s} {'floor PSNR':>11s} "
      f"{'conv MSE':>10s} {'headroom':>9s}")
for size in [12, 24, 48]:
    conv = make_conv(inputs, kernels, ii, kk)
    if size != 24:
        conv = F.interpolate(conv, size=(size, size),
                             mode="area" if size < 24 else "bilinear",
                             **({} if size < 24 else {"align_corners": False}))
    for nname, nf in NORMS.items():
        ta = nf(pool_to(a48, size))
        tb = nf(pool_to(b48, size))
        to = nf(pool_to(o48, size))
        floor = (F.mse_loss(ta, tb) / 2).item()
        cb = F.mse_loss(nf(conv.squeeze(1)), to).item()
        psnr = 10 * np.log10(1.0 / floor) if nname != "zscore" else float("nan")
        print(f"{size:5d} {nname:>8s} {floor:12.6f} {psnr:11.2f} {cb:10.6f} "
              f"{cb/floor:8.0f}x")

print("\nNote: PSNR assumes data range 1.0, so it is meaningful for minmax/robust only.")
print("'headroom' = conv-baseline MSE / floor MSE: the factor a perfect model would gain.")
print("Paste the floor column into train.FLOOR.")

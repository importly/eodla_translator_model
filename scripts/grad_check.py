"""Gradient fidelity: does the surrogate's gradient point where the simulator goes?

An outer model trains through the surrogate, so what it follows is dL/dw and dL/dx. The
true effect of flipping one +-1 pixel is a simulator difference. Per row, R random
readouts L = <r, y> stand in for downstream losses, and for every kernel and input pixel
the surrogate's predicted change, dL/dpixel x (-2 pixel), is compared with the change in
the simulator's noise-free training target. corr and sign near 1: the gradients point
the right way. The same numbers for minmax(conv) - treating the bench as an ideal
convolution - are the baseline the surrogate has to beat. Test MSE cannot see any of
this; README, "Why the kernels are free 81-bit".

    uv run python scripts/grad_check.py runs/pilot3/model.pt [--rows 32]
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import h5py, numpy as np, torch, torch.nn.functional as F

from paths import DATA
from eodla_sim import TorchEODLASim
from data import IN_CH, make_conv, norm_minmax, pool_to, predict
from models import build

CROP, R, BATCH = 144, 8, 64

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--rows", type=int, default=32, help="rows per struct family")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cuda")
cfg = ck["cfg"]
rep = cfg.get("rep", "conv")
model = build(cfg["model"], IN_CH[rep], **cfg.get("model_kw", {})).cuda().eval()
model.load_state_dict(ck["state_dict"])
sim = TorchEODLASim()


def conv(x, w):
    i = torch.arange(len(x), device=x.device)
    return make_conv(x, w, i, i)[:, 0]


def surrogate(x, w):
    return predict(model, rep, x, w, conv(x, w))


def ideal(x, w):
    return norm_minmax(conv(x, w))


def truth(x, w):
    """Noise-free training target, made the way generate.py and Bundle make it."""
    out = []
    for s in range(0, len(x), BATCH):
        full = sim.simulate(x[s:s + BATCH], w[s:s + BATCH], noise=False)
        r0, c0 = (full.shape[-2] - CROP) // 2, (full.shape[-1] - CROP) // 2
        t48 = F.interpolate(full[:, None, r0:r0 + CROP, c0:c0 + CROP], (48, 48), mode="area")
        out.append(norm_minmax(pool_to(t48[:, 0], 24)))
    return torch.cat(out)


def flips(v):
    """(n,) -> (n + 1, n): v itself, then v with each entry negated in turn."""
    return torch.cat([v[None], v[None] * (1 - 2 * torch.eye(len(v), device=v.device))])


def changes(f, x, w, readouts):
    """(R, 81 + 256): each readout's dL/dpixel x (-2 pixel), kernel pixels first."""
    x, w = x.clone().requires_grad_(), w.clone().requires_grad_()
    y = f(x[None], w[None])[0]
    out = []
    for r in readouts:
        gx, gw = torch.autograd.grad((r * y).sum(), (x, w), retain_graph=True)
        out.append(torch.cat([(-2 * gw * w).flatten(), (-2 * gx * x).flatten()]))
    return torch.stack(out)


def agree(t, p):
    """Mean correlation over readouts, and the fraction of pixels whose signs match."""
    tc, pc = t - t.mean(1, keepdim=True), p - p.mean(1, keepdim=True)
    c = (tc * pc).sum(1) / (tc.norm(dim=1) * pc.norm(dim=1) + 1e-12)
    return [c.mean().item(), (t.sign() == p.sign()).float().mean().item()]


d = np.load(DATA / "operands.npz")
names = d["struct_names"]
with h5py.File(DATA / "gen_struct.h5", "r", locking=False) as f:
    fam = d["struct_family"][f["kk"][:]]
    pick = np.sort(np.concatenate([np.where(fam == k)[0][:a.rows] for k in range(len(names))]))
    X = torch.from_numpy(f["input16"][pick]).cuda()
    W = torch.from_numpy(f["kernel9"][pick]).cuda()
fam = fam[pick]
g = torch.Generator(device="cuda").manual_seed(0)

res = {k: [] for k in range(len(names))}
for i in range(len(pick)):
    x, w = X[i], W[i]
    tk = truth(x[None].expand(82, 16, 16), flips(w.flatten()).reshape(-1, 9, 9))
    tx = truth(flips(x.flatten())[1:].reshape(-1, 16, 16), w[None].expand(256, 9, 9))
    readouts = torch.randn(R, 24, 24, device="cuda", generator=g)
    true = torch.einsum("rhw,phw->rp", readouts, torch.cat([tk[1:], tx]) - tk[0])
    row = []
    for fn in (surrogate, ideal):
        pred = changes(fn, x, w, readouts)
        row += agree(true[:, :81], pred[:, :81]) + agree(true[:, 81:], pred[:, 81:])
    res[int(fam[i])].append(row)

print(f"\n{a.ckpt}   {cfg['model']} / {rep}   {a.rows} rows per family, {R} readouts, "
      f"noise-free simulator")
print(f"  {'':8s} {'-------- surrogate --------':>29s}   {'--- ideal conv ---':>18s}")
print(f"  {'family':8s} {'kernel':>7s} {'sign':>5s} {'input':>7s} {'sign':>5s}   "
      f"{'kernel':>7s} {'input':>7s}")
for k, name in enumerate(names):
    m = np.mean(res[k], 0)
    print(f"  {name:8s} {m[0]:7.2f} {m[1]:5.2f} {m[2]:7.2f} {m[3]:5.2f}   {m[4]:7.2f} {m[6]:7.2f}")

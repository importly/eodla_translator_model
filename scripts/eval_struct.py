"""Score saved checkpoints on the structured-kernel split, one line per kernel family.

gen_struct.h5 is test inputs x four kernel families; 'uniform' is ordinary held-out
test kernels, the control. Every row carries two noise draws, so each family's floor
is measured on its own rows. 'vs uniform' is a family's x-floor over the control's:
~1 means the surrogate generalises to that kind of kernel. A second table splits each
family by kernel pixels on. See README.

    uv run python scripts/eval_struct.py runs/v1/model.pt [more.pt ...]
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import h5py, numpy as np, torch

from paths import DATA
from data import IN_CH, norm_minmax, pool_to, predict
from models import build

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
a = ap.parse_args()

d = np.load(DATA / "operands.npz")
names = d["struct_names"]
with h5py.File(DATA / "gen_struct.h5", "r", locking=False) as f:
    x, w, conv, ta, tb = (torch.from_numpy(f[c][:]).cuda()
                          for c in ("input16", "kernel9", "conv24", "target48", "target48b"))
    fam = torch.from_numpy(d["struct_family"][f["kk"][:]]).cuda()
ta, tb = norm_minmax(pool_to(ta, 24)), norm_minmax(pool_to(tb, 24))
floor = ((ta - tb) ** 2).mean((-2, -1)) / 2         # two draws differ by sqrt(2) x noise

for p in a.ckpts:
    ck = torch.load(p, map_location="cuda")
    cfg = ck["cfg"]
    rep = cfg.get("rep", "conv")
    model = build(cfg["model"], IN_CH[rep], **cfg.get("model_kw", {})).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    se = []
    with torch.no_grad():
        for s in range(0, len(ta), 512):
            y = predict(model, rep, x[s:s + 512], w[s:s + 512], conv[s:s + 512])
            se.append(((y - ta[s:s + 512]) ** 2).mean((-2, -1)))
    mse = torch.cat(se)

    xf = [(mse[fam == k].mean() / floor[fam == k].mean()).item() for k in range(len(names))]
    print(f"\n{p}   {cfg['model']} / {rep}")
    print(f"  {'family':8s} {'rows':>6s} {'mse':>9s} {'floor':>9s} {'x floor':>8s} {'vs uniform':>10s}")
    for k, name in enumerate(names):
        m = fam == k
        print(f"  {name:8s} {int(m.sum()):6d} {mse[m].mean():9.6f} {floor[m].mean():9.6f} "
              f"{xf[k]:7.1f}x {xf[k] / xf[0]:10.2f}")

    # same 'vs uniform', split by how many of the 81 kernel pixels are on (rows)
    on = (w > 0).flatten(1).sum(1)
    edges = [0, 10, 20, 30, 40, 50, 60, 70, 82]
    print(f"\n  {'on':8s}" + "".join(f"{n:>14s}" for n in names))
    for lo, hi in zip(edges[:-1], edges[1:]):
        cells = []
        for k in range(len(names)):
            m = (fam == k) & (on >= lo) & (on < hi)
            n = int(m.sum())
            cells.append(f"{mse[m].mean() / floor[m].mean() / xf[0]:.2f} ({n})" if n >= 20 else "")
        print(f"  {f'{lo}-{hi - 1}':8s}" + "".join(f"{c:>14s}" for c in cells))

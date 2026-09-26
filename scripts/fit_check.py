"""Does a checkpoint fit the kernels it trained on? Per kernel pixels-on bin, x floor on
its own training rows against x floor on held-out struct rows of the families it trained
on (the held-out family is left out), both over the struct rows' floor.

Train x floor in a bin close to the 30-49 bins': the model fits that kind of kernel, so a
large held-out gap there is a data problem. Train x floor high as well: the model or its
input can't represent it. README, "If a family fails".

    uv run python scripts/fit_check.py runs/pilot3/model.pt [--rows 200000]
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import h5py, numpy as np, torch

from paths import DATA
from data import IN_CH, norm_minmax, pool_to, predict
from models import build

EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 82]
CHUNK = 20_000

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--rows", type=int, default=200_000, help="train rows the checkpoint saw")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cuda")
cfg = ck["cfg"]
rep = cfg.get("rep", "conv")
model = build(cfg["model"], IN_CH[rep], **cfg.get("model_kw", {})).cuda().eval()
model.load_state_dict(ck["state_dict"])


def score(f, rows):
    """Per-row MSE against target48 and kernel pixels on, CHUNK rows at a time."""
    se, on = [], []
    for s in range(0, rows, CHUNK):
        x, w, conv, t = (torch.from_numpy(f[c][s:min(s + CHUNK, rows)]).cuda()
                         for c in ("input16", "kernel9", "conv24", "target48"))
        t = norm_minmax(pool_to(t, 24))
        with torch.no_grad():
            for b in range(0, len(t), 512):
                y = predict(model, rep, x[b:b + 512], w[b:b + 512], conv[b:b + 512])
                se.append(((y - t[b:b + 512]) ** 2).mean((-2, -1)))
        on.append((w > 0).flatten(1).sum(1))
    return torch.cat(se), torch.cat(on)


d = np.load(DATA / "operands.npz")
held = str(d["held_out"])
with h5py.File(DATA / "gen_train.h5", "r", locking=False) as f:
    tr_se, tr_on = score(f, min(a.rows, f["target48"].shape[0]))
with h5py.File(DATA / "gen_struct.h5", "r", locking=False) as f:
    ho_se, ho_on = score(f, f["target48"].shape[0])
    ta, tb = (norm_minmax(pool_to(torch.from_numpy(f[c][:]).cuda(), 24))
              for c in ("target48", "target48b"))
    trained = torch.from_numpy(d["struct_names"][d["struct_family"][f["kk"][:]]] != held).cuda()
floor = ((ta - tb) ** 2).mean((-2, -1)) / 2         # two draws differ by sqrt(2) x noise

print(f"\n{a.ckpt}   {cfg['model']} / {rep}   {len(tr_se)} train rows, {held} left out")
print(f"  {'on':8s} {'train rows':>10s} {'train x':>8s} {'held-out rows':>13s} {'held-out x':>10s} {'gap':>6s}")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    mt = (tr_on >= lo) & (tr_on < hi)
    mh = trained & (ho_on >= lo) & (ho_on < hi)
    if mt.sum() < 20 or mh.sum() < 20:
        continue
    fl = floor[mh].mean()
    xt, xh = (tr_se[mt].mean() / fl).item(), (ho_se[mh].mean() / fl).item()
    print(f"  {f'{lo}-{hi - 1}':8s} {int(mt.sum()):10d} {xt:7.1f}x {int(mh.sum()):13d} "
          f"{xh:9.1f}x {xh / xt:6.2f}")

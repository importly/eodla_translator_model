"""Plot rows of a gen_<split>.h5 - every panel is stored in the file except the two
normalisations.

    uv run python scripts/show_row.py --rows 0 1 2 3 --split test
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import h5py, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import ROOT, DATA, OUT
from data import pool_to, norm_minmax
from train import FLOOR

CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
INK, MUTED = "#1f2933", "#6b7280"

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--split", default="train", choices=["train", "val", "test", "struct"])
ap.add_argument("--out", default=None)
A = ap.parse_args()

with h5py.File(DATA / f"gen_{A.split}.h5", "r", locking=False) as f:
    n_total = f["target48"].shape[0]
    cols = []
    for r in A.rows:
        d = {k: f[k][r] for k in ("input16", "kernel9", "conv24", "target48", "label", "ii", "kk")}
        d["convn"] = norm_minmax(torch.from_numpy(d["conv24"])[None])[0].numpy()
        d["t24"] = norm_minmax(pool_to(torch.from_numpy(d["target48"])[None], 24))[0].numpy()
        d["err"] = np.abs(d["convn"] - d["t24"])
        d["mse"] = float(((d["convn"] - d["t24"]) ** 2).mean())
        cols.append(d)

floor = FLOOR[(24, "minmax")]

ROWS = [   # row label, key, cmap, caption
    ("input x\n16x16", "input16", "gray", lambda d: "{-1, +1}"),
    ("kernel w\n9x9", "kernel9", "gray", lambda d: "{-1, +1}"),
    ("conv24\n(raw, stored)", "conv24", "gray", lambda d: f"[{d['conv24'].min():.0f}, {d['conv24'].max():.0f}]"),
    ("target48\n(raw, stored)", "target48", "gray", lambda d: f"[{d['target48'].min():.0f}, {d['target48'].max():.0f}]"),
    ("MODEL INPUT\nminmax(conv24)", "convn", "gray", lambda d: "rep='conv'  (24x24)"),
    ("MODEL TARGET\nminmax(pool(t48))", "t24", "gray", lambda d: "24x24, [0,1]"),
    ("|input - target|", "err", "inferno", lambda d: f"mse {d['mse']:.4f}  ({d['mse']/floor:.0f}x floor)"),
]

fig, axes = plt.subplots(len(ROWS), len(cols),
                         figsize=(2.6 * len(cols) + 1.6, 2.5 * len(ROWS)), squeeze=False)
fig.patch.set_facecolor("white")

for j, d in enumerate(cols):
    cls, source, src = d["label"]
    name = (f"CIFAR '{CIFAR[int(cls)]}' #{int(src)}" if source == 1 else
            f"EMNIST digit #{int(src)}" if source == 0 else f"random pattern #{int(src)}")
    axes[0][j].set_title(f"row {A.rows[j]}   ii={int(d['ii'])} kk={int(d['kk'])}\n{name}",
                         fontsize=9.5, color=INK, pad=22)
    for i, (label, key, cmap, cap) in enumerate(ROWS):
        ax = axes[i][j]
        ax.imshow(d[key], cmap=cmap, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.5, 1.02, cap(d), fontsize=8, color=MUTED,
                ha="center", va="bottom", transform=ax.transAxes)
        if j == 0:
            ax.set_ylabel(label, fontsize=9.5, color=INK, rotation=0,
                          ha="right", va="center", labelpad=14)

fig.suptitle(f"gen_{A.split}.h5  ({n_total} rows)   -   noise floor {floor}",
             fontsize=11, color=INK, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.975])

out = ROOT / A.out if A.out else OUT / f"{A.split}_rows.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130, facecolor="white")
print(f"wrote {out}")

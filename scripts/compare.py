"""The comparison for the write-up: the mentor's conv-only spec, then why convikscale.

Every entry in models.TRACK trained with one recipe (the v3_l1g1 config) on the same data,
so the table is like-for-like:

    conv/linear       one affine map, 576 -> 576. No nonlinearity anywhere.
    conv/cnn          plain conv stack - no residual connections, no pooling, no skips.
    conv/linconv      the two summed; blend starts at 0, so it begins as exactly `linear`.
    conv/resunet-96   a strong net on the same input: where conv-only tops out.
    convikscale/...   the same net also shown x and w - the chosen surrogate.

One run per call with --only, so each can go on its own GPU; without it, every missing
run in turn. Results: out/compare/<name>.json, checkpoints: runs/compare/<name>.pt.
The table of everything finished is printed at the end.

    uv run python scripts/compare.py --only conv/linear [--train_n 60000 --epochs 20]
"""
import sys, pathlib, json, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import Bundle
from models import TRACK
from train import run, save
from paths import RUNS, OUT

RECIPE = dict(loss="l1", w_grad=1.0, lr=1e-3, wd=1e-4, batch=256, clip=1.0, seed=0)

ap = argparse.ArgumentParser()
ap.add_argument("--only", choices=[n for n, _ in TRACK])
ap.add_argument("--train_n", type=int, default=None, help="default: all training rows")
ap.add_argument("--epochs", type=int, default=40)
A = ap.parse_args()

res_dir = OUT / "compare"
res_dir.mkdir(parents=True, exist_ok=True)
path = lambda n: res_dir / f"{n.replace('/', '_')}.json"
todo = [(n, ov) for n, ov in TRACK if (A.only in (None, n)) and not path(n).exists()]

if todo:
    b = Bundle(max_train=A.train_n)
    for name, ov in todo:
        print(f"\n=== {name} ===", flush=True)
        cfg = dict(RECIPE, epochs=A.epochs, **ov,
                   ckpt=str(RUNS / "compare" / f"{name.replace('/', '_')}.pt"))
        r, _ = run(b, cfg, log_every=5)
        r["name"], r["n_train"] = name, b.splits["train"]["t"].shape[0]
        save(r, path(name))

print(f"\n  {'':24s} {'test mse':>9s} {'x floor':>8s} {'gap':>5s} {'params':>7s} {'rows':>8s}")
for name, _ in TRACK:
    if path(name).exists():
        r = json.load(open(path(name)))
        print(f"  {name:24s} {r['test']['mse']:9.6f} {r['test_over_floor']:7.1f}x "
              f"{r['gap_train_val']:5.2f} {r['params']/1e6:6.2f}M {r['n_train']:8d}")

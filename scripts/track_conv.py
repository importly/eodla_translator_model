"""The conv-only track: the mentor's spec ("GPU convolutions in, optical out").

conv24 in, pooled target out - these models never see the input image or the kernel.
Three capacity classes, so the gaps between them say how much of the task is linear:

    linear    one affine map, 576 -> 576. No nonlinearity anywhere.
    cnn       plain conv stack - no residual connections, no pooling, no skips.
    linconv   the two summed; blend starts at 0, so it begins as exactly `linear`.

Plus two reference points: a strong net on the same input, and conv4 - the four unsigned
convolutions the bench actually forms - as a richer but still convolution-only input.
Same data, epochs and loss as pipeline.py, so numbers are like-for-like with
out/pipe_arch.json.

Resumable per run; appends events to out/pipeline.log so the monitor sees them.
"""
import sys, pathlib, json, time, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

from data import Bundle
from models import CONV_TRACK
from train import run, save

from paths import RUNS, OUT
OUT.mkdir(exist_ok=True)
PATH = OUT / "pipe_convtrack.json"

BASE = dict(loss="l1", w_grad=0.5, lr=2e-3, wd=1e-4, batch=256, epochs=20, amp=True,
            clip=1.0, augment=False, seed=0)

GRID = CONV_TRACK

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--list", action="store_true", help="show the grid and what is already done, then exit")
ap.add_argument("--train_n", type=int, default=60000)
ap.add_argument("--epochs", type=int, default=20)
A = ap.parse_args()
BASE["epochs"] = A.epochs

rows = json.load(open(PATH)) if PATH.exists() else []
done = {r["name"] for r in rows}
todo = [(n, ov) for n, ov in GRID if n not in done]
if A.list:
    for n, _ in GRID:
        print(("done  " if n in done else "todo  ") + n)
    sys.exit(0)
print(f"\n{'#'*78}\n### stage convtrack  ({len(todo)} of {len(GRID)} runs to do, {A.epochs} ep, train_n {A.train_n})\n{'#'*78}", flush=True)

if todo:
    b = Bundle(size=24, tnorm="minmax", max_train=A.train_n)
    t0 = time.time()
    for name, ov in todo:
        print(f"\n=== convtrack/{name} ===", flush=True)
        cfg = dict(BASE, **ov)
        cfg["ckpt"] = str(RUNS / "sweep" / f"convtrack_{name.replace('/', '_')}.pt")
        r, _ = run(b, cfg, log_every=5)
        r["name"] = name
        r["n_train"] = b.splits["train"]["t"].shape[0]
        rows.append(r)
        save(rows, PATH)
        torch.cuda.empty_cache()
    print(f"\nconvtrack stage: {(time.time() - t0) / 60:.1f} min", flush=True)

rows.sort(key=lambda r: r["test"]["mse"])
print(f"\n>>> convtrack winner: {rows[0]['name']}  test mse {rows[0]['test']['mse']:.6f}  "
      f"({rows[0]['test_over_floor']:.1f}x floor)", flush=True)
ref = {"conv/resunet-64 (rep stage)": 0.011201, "convikscale/resunet-96 (arch winner)": 0.001143}
for r in rows:
    print(f"    {r['name']:>22s}  {r['test']['mse']:.6f}  {r['test_over_floor']:6.1f}x  "
          f"gap {r['gap_train_val']:.1f}x  {r['params']/1e6:.2f}M  {r['minutes']:.1f} min")
for k, v in ref.items():
    print(f"    {k:>22s}  {v:.6f}  (reference)")

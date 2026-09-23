"""Sequential, resumable sweep chain. Each stage picks its winner and feeds the next.

    rep -> arch -> loss -> hparam -> datascale -> final

Stage results go to out/pipe_<stage>.json; runs already present in it are skipped,
so the whole thing can be re-launched after an interruption. out/best_cfg.json is
rewritten after every stage.

Usage: uv run python -u scripts/pipeline.py [--train_n 60000] [--epochs 20] [--stages rep,arch,...]
"""
import sys, pathlib, argparse, json, time, copy
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch

from data import Bundle
from models import FULL_TRACK, FULL_REPS
from train import run, save

from paths import RUNS, OUT
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--train_n", type=int, default=60000, help="samples per sweep run")
ap.add_argument("--epochs", type=int, default=20, help="epochs per sweep run")
ap.add_argument("--final_epochs", type=int, default=40)
ap.add_argument("--stages", type=str, default="rep,arch,loss,hparam,datascale,final")
A = ap.parse_args()
STAGES = A.stages.split(",")

BEST_PATH = OUT / "best_cfg.json"
if BEST_PATH.exists():
    best = json.load(open(BEST_PATH))
    print(f"resuming from best_cfg: {json.dumps(best)}")
else:
    best = dict(model="resunet", model_kw=dict(base=64), rep="convik", loss="l1", w_grad=0.5,
                lr=2e-3, wd=1e-4, batch=256, amp=True, clip=1.0, augment=False, seed=0)

T0 = time.time()
timings = {}


def cfg_with(**kw):
    """Copy of the current best config with overrides applied (model_kw replaced whole)."""
    c = copy.deepcopy(best)
    mk = kw.pop("model_kw", None)
    c.update(kw)
    if mk is not None:
        c["model_kw"] = mk
    return c


def stage(name, grid, bundle_kw=None, epochs=None, pick_key=("val", "mse")):
    """grid: list of (label, cfg_overrides). Returns winner label + rows."""
    global best
    path = OUT / f"pipe_{name}.json"
    if name not in STAGES:
        return None
    rows = json.load(open(path)) if path.exists() else []
    done_labels = {r["name"] for r in rows}
    todo = [(l, ov) for l, ov in grid if l not in done_labels]
    if not todo:
        print(f"\n### {name}: already done, {len(rows)} rows (skipping)")
    else:
        print(f"\n{'#'*78}\n### stage {name}  ({len(todo)} of {len(grid)} runs to do, "
              f"{A.epochs if epochs is None else epochs} ep, "
              f"train_n {bundle_kw.get('max_train') if bundle_kw else A.train_n})\n{'#'*78}", flush=True)
        ts = time.time()
        bk = dict(size=24, tnorm="minmax", max_train=A.train_n)
        if bundle_kw:
            bk.update(bundle_kw)
        b = Bundle(**bk)
        for label, ov in todo:
            print(f"\n=== {name}/{label} ===", flush=True)
            c = cfg_with(**ov)
            c["epochs"] = A.epochs if epochs is None else epochs
            c["ckpt"] = str(RUNS / "sweep" / f"{name}_{label}.pt")
            r, _ = run(b, c, log_every=5)
            r["name"] = label
            r["n_train"] = b.splits["train"]["t"].shape[0]
            rows.append(r)
            save(rows, path)                       # partial progress survives a crash
            torch.cuda.empty_cache()
        del b
        torch.cuda.empty_cache()
        timings[name] = (time.time() - ts) / 60
    rows.sort(key=lambda r: r[pick_key[0]][pick_key[1]])
    win = rows[0]
    print(f"\n>>> {name} winner: {win['name']}  val mse {win['val']['mse']:.6f}  "
          f"test mse {win['test']['mse']:.6f}  ({win['test_over_floor']:.1f}x floor)", flush=True)
    for r in rows:
        print(f"    {r['name']:>18s}  val {r['val']['mse']:.6f}  test {r['test']['mse']:.6f}  "
              f"{r['test_over_floor']:6.1f}x  "
              f"gap {r['gap_train_val']:.1f}x  {r['params']/1e6:.2f}M  {r['minutes']:.1f} min")
    # adopt winner's config fields
    wc = dict(win["cfg"]); wc["model_kw"] = win.get("model_kw", {})
    for k in ("model", "model_kw", "rep", "loss", "w_grad", "lr", "wd", "augment"):
        if k in wc:
            best[k] = wc[k]
    best.pop("epochs", None); best.pop("limit", None)
    json.dump(best, open(BEST_PATH, "w"), indent=1)
    return win["name"], rows


# ------------------------------------------------------------------ stages
stage("rep", FULL_REPS)

stage("arch", FULL_TRACK)

stage("loss", [
    ("l1", dict(loss="l1", w_grad=0.0)),
    ("l1+grad0.5", dict(loss="l1", w_grad=0.5)),
    ("l1+grad1.0", dict(loss="l1", w_grad=1.0)),
    ("l2", dict(loss="l2", w_grad=0.0)),
    ("l2+grad0.5", dict(loss="l2", w_grad=0.5)),
])

stage("hparam", [
    ("lr1e-3", dict(lr=1e-3, wd=1e-4)),
    ("lr2e-3", dict(lr=2e-3, wd=1e-4)),
    ("lr4e-3", dict(lr=4e-3, wd=1e-4)),
    ("lr2e-3-wd1e-3", dict(lr=2e-3, wd=1e-3)),
    ("lr2e-3-aug", dict(lr=2e-3, wd=1e-4, augment=True)),
])

# data scaling: separate bundles per size, so handled inline
if "datascale" in STAGES:
    path = OUT / "pipe_datascale.json"
    rows = json.load(open(path)) if path.exists() else []
    done_n = {r["n_train"] for r in rows}
    todo_n = [n for n in [16000, 60000, 150000, 400000, 1000000, 2000000] if n not in done_n]
    if not todo_n:
        print("\n### datascale: already done (skipping)")
    else:
        print(f"\n{'#'*78}\n### stage datascale ({len(todo_n)} sizes to do)\n{'#'*78}", flush=True)
        ts = time.time()
        for n in todo_n:
            print(f"\n=== datascale/n{n} ===", flush=True)
            b = Bundle(size=24, tnorm="minmax", max_train=n)
            got = b.splits["train"]["t"].shape[0]
            c = cfg_with(); c["epochs"] = A.epochs
            c["ckpt"] = str(RUNS / "sweep" / f"datascale_n{got}.pt")
            r, _ = run(b, c, log_every=5)
            r["name"] = f"n{got}"; r["n_train"] = got
            rows.append(r); save(rows, path)
            del b; torch.cuda.empty_cache()
            if got < n:
                break
        timings["datascale"] = (time.time() - ts) / 60
        print("\n>>> datascale:")
        for r in rows:
            print(f"    {r['name']:>10s}  test {r['test']['mse']:.6f}  {r['test_over_floor']:6.1f}x  "
                  f"gap {r['gap_train_val']:.1f}x")

# final: full data, long schedule, two seeds
if "final" in STAGES:
    path = OUT / "pipe_final.json"
    if path.exists():
        print("\n### final: already done (skipping)")
    else:
        print(f"\n{'#'*78}\n### stage final  ({A.final_epochs} ep, all data)\n{'#'*78}", flush=True)
        ts = time.time()
        b = Bundle(size=24, tnorm="minmax")
        rows = []
        for seed in (0, 1):
            print(f"\n=== final/seed{seed} ===", flush=True)
            c = cfg_with(seed=seed); c["epochs"] = A.final_epochs
            c["ckpt"] = str(RUNS / f"final_seed{seed}.pt")
            r, model = run(b, c, log_every=2)
            r["name"] = f"seed{seed}"; r["n_train"] = b.splits["train"]["t"].shape[0]
            rows.append(r); save(rows, path)
            RUNS.mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "cfg": c, "metrics": r},
                       RUNS / f"final_seed{seed}.pt")
            del model; torch.cuda.empty_cache()
        timings["final"] = (time.time() - ts) / 60
        print("\n>>> final:")
        for r in rows:
            print(f"    {r['name']}  test {r['test']['mse']:.6f}  psnr {r['test']['psnr']:.2f}  "
                  f"{r['test_over_floor']:.1f}x floor  gap {r['gap_train_val']:.1f}x")

print(f"\nstage minutes: {json.dumps({k: round(v, 1) for k, v in timings.items()})}")
print(f"total {(time.time()-T0)/60:.1f} min")
print(f"best_cfg: {json.dumps(best)}")
print("PIPELINE DONE")

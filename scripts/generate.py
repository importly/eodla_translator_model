"""Stage 2: (input, kernel) pairs -> simulator -> data/gen_<split>.h5

One row is one complete sample. Resumable: re-run and it tops the file up to --n,
re-seeding from the row count.

    uv run python scripts/generate.py --split train --n 2000000
    uv run python scripts/generate.py --split val   --n 10000
    uv run python scripts/generate.py --split test  --n 10000
    uv run python scripts/generate.py --split struct --n 10000

struct is test inputs x struct_kernels (held-out kernel families, see README), with a
second noise draw per row in target48b so each family's floor can be measured.
"""
import sys, pathlib, time, argparse, subprocess, datetime, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import h5py, numpy as np, torch, torch.nn.functional as F

from paths import ROOT, DATA
from eodla_sim import TorchEODLASim, C, B
from data import make_conv

CROP, SIZE, FLUSH = 144, 48, 8192
REPORT_EVERY = 30.0                       # seconds; a 2M run flushes ~240 times
COLS = {"input16": (16, 16), "kernel9": (9, 9), "conv24": (24, 24),
        "target48": (SIZE, SIZE), "label": (3,), "ii": (), "kk": ()}

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train", choices=["train", "val", "test", "struct"])
ap.add_argument("--n", type=int, default=400_000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

struct = a.split == "struct"
d = np.load(DATA / "operands.npz")
inputs = torch.from_numpy(d["inputs"]).cuda()
kernels = torch.from_numpy(d["struct_kernels" if struct else "kernels"]).cuda()
labels = torch.from_numpy(d["labels"]).cuda()
n_train, n_val = int(d["n_train"]), int(d["n_val"])
lo, hi = {"train":  (0, n_train),
          "val":    (n_train, n_train + n_val),
          "test":   (n_train + n_val, len(inputs)),
          "struct": (n_train + n_val, len(inputs))}[a.split]
klo, khi = (0, len(kernels)) if struct else (lo, hi)
if struct:
    COLS["target48b"] = (SIZE, SIZE)

out = DATA / f"gen_{a.split}.h5"
f = h5py.File(out, "a")
if "target48" in f and f["target48"].shape[0] and any(c not in f for c in COLS):
    sys.exit(f"{out} is an old-format file ({f['target48'].shape[0]} rows, no input16). "
             "Move it aside rather than appending to it.")
for name, shp in COLS.items():
    if name not in f:
        f.create_dataset(name, shape=(0, *shp), maxshape=(None, *shp),
                         dtype=np.int32 if name in ("ii", "kk") else np.float32,
                         chunks=(256, *shp) if shp else (4096,))

if "kernel_scheme" not in f.attrs:
    # shapes alone cannot tell two datasets apart; record how this one was made
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    f.attrs.update({"kernel_scheme": "struct" if struct else "free81", "input_sources": "cifar45/emnist45/noise10",
                    "split": a.split, "seed": a.seed, "pool_lo": lo, "pool_hi": hi,
                    "created": datetime.datetime.now().isoformat(timespec="seconds"),
                    "git_commit": commit or "uncommitted",
                    "sim_grid": 1080, "sim_camera": 346, "sim_stored_rows": 260,
                    "sim_camera_model": f"events, combined, C={C} B={B}",
                    "sim_dmd": "binary, footprint only",
                    "sim_input_grid_px": 300, "sim_kernel_grid_px": 168,
                    "crop": CROP, "target_size": SIZE})

done = f["target48"].shape[0]
todo = max(0, a.n - done)
need = todo * 4 * sum(int(np.prod(s)) for s in COLS.values())     # all 4-byte columns
free = shutil.disk_usage(DATA).free
if free < need * 1.05:
    f.close()
    sys.exit(f"need ~{need/1e9:.1f} GB for {todo} more rows, {free/1e9:.1f} GB free")
print(f"{a.split}: operands [{lo},{hi})   have {done}, making {todo}  (~{need/1e9:.1f} GB)   "
      f"[{f.attrs['kernel_scheme']}, {f.attrs['input_sources']}]")

rng = np.random.default_rng([a.seed, done, lo])
gen = torch.Generator(device="cuda"); gen.manual_seed(a.seed * 1_000_003 + done + lo)
sim = TorchEODLASim()


def sim48(x, w):
    """Camera frame -> centre crop CROP -> area-pool to SIZE."""
    full = sim.simulate(x, w, gen=gen)
    r0 = (full.shape[-2] - CROP) // 2
    c0 = (full.shape[-1] - CROP) // 2
    crop = full[..., r0:r0 + CROP, c0:c0 + CROP].unsqueeze(1)
    return F.interpolate(crop, (SIZE, SIZE), mode="area").squeeze(1).cpu().numpy()


t0, start, last = time.time(), done, 0.0
buf = {k: [] for k in COLS}

while done < a.n:
    b = min(a.batch, a.n - done)
    ii, kk = rng.integers(lo, hi, b), rng.integers(klo, khi, b)
    ti = torch.as_tensor(ii, device="cuda")
    tk = torch.as_tensor(kk, device="cuda")

    buf["input16"].append(inputs[ii].cpu().numpy())
    buf["kernel9"].append(kernels[kk].cpu().numpy())
    buf["conv24"].append(make_conv(inputs, kernels, ti, tk)[:, 0].cpu().numpy())
    buf["target48"].append(sim48(inputs[ii], kernels[kk]))
    if struct:
        buf["target48b"].append(sim48(inputs[ii], kernels[kk]))
    buf["label"].append(labels[ii].cpu().numpy())
    buf["ii"].append(ii.astype(np.int32))
    buf["kk"].append(kk.astype(np.int32))
    done += b

    if len(buf["ii"]) * a.batch >= FLUSH or done >= a.n:
        n0 = f["target48"].shape[0]
        for name in COLS:
            arr = np.concatenate(buf[name])
            f[name].resize(n0 + len(arr), axis=0)
            f[name][n0:] = arr
            buf[name] = []
        f.flush()
        if time.time() - last >= REPORT_EVERY or done >= a.n:
            last = time.time()
            rate = (done - start) / (last - t0)
            print(f"  {done}/{a.n}   {rate:.0f}/s   eta {(a.n - done) / rate / 60:.0f} min",
                  flush=True)

f.close()
print(f"{out}   {out.stat().st_size / 1e9:.1f} GB")

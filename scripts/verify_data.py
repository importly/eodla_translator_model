"""One-off audit of a finished generation run. Exits non-zero on any FAIL.

    uv run python scripts/verify_data.py
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import h5py, numpy as np, torch

from paths import DATA
from data import make_conv

COLS = {"input16": (16, 16), "kernel9": (9, 9), "conv24": (24, 24),
        "target48": (48, 48), "label": (3,), "ii": (), "kk": ()}
FRAC = {"CIFAR": (1, 0.45), "EMNIST": (0, 0.45), "noise": (2, 0.10)}
CHUNK = 20_000

ap = argparse.ArgumentParser()
ap.add_argument("--conv-sample", type=int, default=2000)
a = ap.parse_args()

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


d = np.load(DATA / "operands.npz")
op_in, op_ker, op_lab = torch.from_numpy(d["inputs"]), torch.from_numpy(d["kernels"]), d["labels"]
st_ker = torch.from_numpy(d["struct_kernels"])
n_train, n_val = int(d["n_train"]), int(d["n_val"])
t0 = n_train + n_val
SPLITS = {   # input pool [lo, hi), kernel table, kernel pool [klo, khi)
    "train":  (0, n_train, op_ker, 0, n_train),
    "val":    (n_train, t0, op_ker, n_train, t0),
    "test":   (t0, len(op_in), op_ker, t0, len(op_in)),
    "struct": (t0, len(op_in), st_ker, 0, len(st_ker)),
}
print(f"operands.npz: {len(op_in)}   " +
      "  ".join(f"{k} [{lo},{hi})" for k, (lo, hi, *_) in SPLITS.items()))

patterns = {}

for split, (lo, hi, ker, klo, khi) in SPLITS.items():
    struct = split == "struct"
    cols = {**COLS, "target48b": (48, 48)} if struct else COLS
    path = DATA / f"gen_{split}.h5"
    print(f"\n{path.name}")
    with h5py.File(path, "r", locking=False) as f:
        scheme = f.attrs.get("kernel_scheme", "")
        check("provenance attrs", bool(scheme),
              f"{scheme} {f.attrs.get('input_sources','')} {f.attrs.get('git_commit','')}"
              if scheme else "none - file predates stamping")
        want_scheme = "struct" if struct else "free81"
        check(f"kernel_scheme is {want_scheme}", scheme == want_scheme, f"got {scheme!r}")

        missing = [c for c in cols if c not in f]
        check("all columns present", not missing, f"missing {missing}" if missing else "")
        if missing:
            continue
        n = f["target48"].shape[0]
        check("row counts agree", len({f[c].shape[0] for c in cols}) == 1, f"{n} rows")
        check("column shapes", all(f[c].shape[1:] == s for c, s in cols.items()))

        ii, kk = f["ii"][:], f["kk"][:]
        check("ii within pool", ii.min() >= lo and ii.max() < hi, f"[{ii.min()}, {ii.max()}]")
        check("kk within pool", kk.min() >= klo and kk.max() < khi, f"[{kk.min()}, {kk.max()}]")

        s = np.sort(np.random.default_rng(0).choice(n, min(a.conv_sample, n), replace=False))
        x, w = f["input16"][s], f["kernel9"][s]
        check("input16 == operands[ii]", np.array_equal(x, op_in[ii[s]].numpy()))
        check("kernel9 == operands[kk]", np.array_equal(w, ker[kk[s]].numpy()))
        check("label == operands[ii]", np.array_equal(f["label"][s], op_lab[ii[s]]))

        want = make_conv(op_in, ker, torch.from_numpy(ii[s].astype(np.int64)),
                         torch.from_numpy(kk[s].astype(np.int64)))[:, 0].numpy()
        err = np.abs(f["conv24"][s] - want).max()
        check("conv24 == conv2d(x, w)", err < 1e-3, f"max abs err {err:.2e}")

        # the OG 6x6 scheme forces rows/cols (0,1) (3,4) (6,7) equal in 100% of kernels.
        # struct kernels are smooth on purpose, so they are exempt
        if not struct:
            dup = float(np.mean([(w[:, i, :] == w[:, j, :]).all(1).mean()
                                 for i, j in ((0, 1), (3, 4), (6, 7))]))
            check("kernels use all 81 dof", dup < 0.05,
                  f"forced-equal rows {dup*100:.2f}%  (6x6 = 100%, chance {2**-9*100:.2f}%)")

        for t in [c for c in cols if c.startswith("target")]:
            bad = dead = 0
            for s0 in range(0, n, CHUNK):
                blk = f[t][s0:s0 + CHUNK]
                bad += int((~np.isfinite(blk)).sum())
                dead += int((blk.std(axis=(1, 2)) == 0).sum())
            check(f"{t} all finite", bad == 0, f"{bad} non-finite")
            check(f"no constant {t}", dead == 0, f"{dead} flat rows")

        # n pairs from m = inputs x kernels possibilities: expected collisions ~ n**2/2m
        expect = n * n / (2 * (hi - lo) * (khi - klo))
        uniq = len(np.unique(np.stack([ii, kk], 1), axis=0))
        check("pair repeats ~ chance", n - uniq <= max(10, 3 * expect),
              f"{n - uniq} repeats, ~{expect:.0f} expected")

        src = f["label"][:, 1].astype(int)
        for name, (code, want_f) in FRAC.items():
            got = int((src == code).sum()) / n
            tol = 4 * np.sqrt(want_f * (1 - want_f) / n)
            check(f"{name} ~{want_f:.0%}", abs(got - want_f) <= tol,
                  f"{got*100:.1f}%  (+-{tol*100:.1f}%)")

        patterns[split] = (
            {r.astype(np.int8).tobytes() for r in np.unique(f["input16"][:], axis=0)},
            {r.astype(np.int8).tobytes() for r in np.unique(f["kernel9"][:], axis=0)},
        )

print("\ncross-split")
if SPLITS.keys() <= patterns.keys():
    tr_x, tr_w = patterns["train"]
    for split in ("val", "test", "struct"):
        vx, vw = patterns[split]
        check(f"train/{split} inputs disjoint", not (tr_x & vx), f"{len(tr_x & vx)} shared")
        check(f"train/{split} kernels disjoint", not (tr_w & vw), f"{len(tr_w & vw)} shared")

print(f"\n{'ALL CHECKS PASSED' if not fails else f'{len(fails)} FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)

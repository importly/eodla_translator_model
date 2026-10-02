"""converting old data"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import h5py, numpy as np, torch, torch.nn.functional as F

from paths import DATA
from data import make_conv

CROP, SIZE, CHUNK = 144, 48, 1000

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(DATA / "simulated-data-v2.h5"))
a = ap.parse_args()

out = DATA / "old_v2"
out.mkdir(exist_ok=True)
names = ("train", "val", "test")
if any((out / f"gen_{n}.h5").exists() for n in names):
    sys.exit(f"{out} already has gen_*.h5 - move them aside first")

with h5py.File(a.src, "r", locking=False) as f:
    w_all = f["kernel"][:]
    _, kid = np.unique(w_all.reshape(len(w_all), -1), axis=0, return_inverse=True)
    kid = kid.ravel()
    nk = kid.max() + 1
    order = np.random.default_rng(0).permutation(nk)
    split = np.empty(nk, int)
    split[order[:int(.8 * nk)]], split[order[int(.8 * nk):int(.9 * nk)]], split[order[int(.9 * nk):]] = 0, 1, 2

    for s, name in enumerate(names):
        rows = np.where(split[kid] == s)[0]
        with h5py.File(out / f"gen_{name}.h5", "w") as g:
            for c, shp, dt in (("input16", (16, 16), "f4"), ("kernel9", (9, 9), "f4"),
                               ("conv24", (24, 24), "f4"), ("target48", (SIZE, SIZE), "f4"),
                               ("label", (3,), "f4"), ("ii", (), "i4"), ("kk", (), "i4")):
                g.create_dataset(c, (len(rows), *shp), dt)
            for c0 in range(0, len(rows), CHUNK):
                r = rows[c0:c0 + CHUNK]
                x, w = torch.from_numpy(f["input"][r]), torch.from_numpy(f["kernel"][r])
                full = torch.from_numpy(f["output-full"][r])
                r0, k0 = (full.shape[-2] - CROP) // 2, (full.shape[-1] - CROP) // 2
                t48 = F.interpolate(full[:, None, r0:r0 + CROP, k0:k0 + CROP], (SIZE, SIZE), mode="area")
                i = torch.arange(len(r))
                sl = slice(c0, c0 + len(r))
                g["input16"][sl], g["kernel9"][sl] = x.numpy(), w.numpy()
                g["conv24"][sl] = make_conv(x, w, i, i)[:, 0].numpy()
                g["target48"][sl] = t48[:, 0].numpy()
                g["label"][sl], g["ii"][sl], g["kk"][sl] = f["label"][r], r, kid[r]
            g.attrs["source"] = a.src
            g.attrs["split"] = "by kernel pattern, 80/10/10, seed 0"
        print(f"{name:5s} {len(rows):6d} rows, {len(np.unique(kid[rows]))} kernels -> {out / f'gen_{name}.h5'}")

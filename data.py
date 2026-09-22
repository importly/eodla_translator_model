"""Data assembly, entirely GPU-resident.

Each split is one self-contained file, data/gen_<split>.h5, whose rows hold the whole
sample. Only two things are derived at load, both per-sample and reversible: target48
is pooled and normalised, and build_input min-max normalises conv24.
"""
import pathlib

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from paths import DATA


def norm_minmax(t):
    lo = t.amin((-2, -1), keepdim=True)
    hi = t.amax((-2, -1), keepdim=True)
    return (t - lo) / (hi - lo + 1e-12)


def norm_zscore(t):
    return (t - t.mean((-2, -1), keepdim=True)) / (t.std((-2, -1), keepdim=True) + 1e-12)


def norm_robust(t, q=0.02):
    """Percentile scaling - one noisy extreme pixel cannot set the range."""
    flat = t.flatten(-2)
    lo = flat.quantile(q, dim=-1, keepdim=True).unsqueeze(-1)
    hi = flat.quantile(1 - q, dim=-1, keepdim=True).unsqueeze(-1)
    return (t - lo) / (hi - lo + 1e-12)


NORMS = {"minmax": norm_minmax, "zscore": norm_zscore, "robust": norm_robust}


def pool_to(t48, size):
    """48 -> {48, 24, 16, 12} by exact integer area pooling."""
    if size == t48.shape[-1]:
        return t48
    return F.avg_pool2d(t48.unsqueeze(1), t48.shape[-1] // size).squeeze(1)


def make_conv(inputs, kernels, ii, kk):
    """Full linear convolution, per-sample, via grouped conv -> (B,1,24,24)."""
    x = inputs[ii]
    w = kernels[kk].unsqueeze(1).flip(-1, -2)
    B = x.shape[0]
    return F.conv2d(x.reshape(1, B, 16, 16), w, padding=8, groups=B).reshape(B, 1, 24, 24)


def pad_to(a, size):
    """Centre-pad a small map into the 24x24 conv grid, preserving alignment."""
    d = size - a.shape[-1]
    return F.pad(a, (d // 2, d - d // 2, d // 2, d - d // 2))


def build_input(rep, x, w, conv, size=24):
    """Model input from one row's stored arrays. Nothing is looked up or recomputed."""
    conv = conv.unsqueeze(1)
    lo = conv.amin((-2, -1), keepdim=True)
    hi = conv.amax((-2, -1), keepdim=True)
    convn = (conv - lo) / (hi - lo + 1e-12)

    if rep == "conv":
        return convn
    if rep == "conv4":
        # the four unsigned convolutions the bench forms and subtracts. Their signed
        # sum is the plain conv, so summing loses information; all four keeps it.
        px = x.clamp(min=0); nx = 1.0 - px
        pw = w.clamp(min=0); nw = 1.0 - pw
        B = px.shape[0]
        chans = [F.conv2d(xi.reshape(1, B, 16, 16), wi.unsqueeze(1).flip(-1, -2),
                          padding=8, groups=B).reshape(B, 1, 24, 24)
                 for xi, wi in ((px, pw), (px, nw), (nx, pw), (nx, nw))]
        c4 = torch.cat(chans, 1)
        return c4 / (c4.amax((-3, -2, -1), keepdim=True) + 1e-12)   # joint scale
    if rep == "convscale":
        return torch.cat([convn, (hi - lo).clamp_min(1e-6).log().expand_as(convn) / 10.0], 1)

    xi, xk = pad_to(x.unsqueeze(1), size), pad_to(w.unsqueeze(1), size)
    if rep == "ik":
        return torch.cat([xi, xk], 1)
    if rep == "convik":
        return torch.cat([convn, xi, xk], 1)
    if rep == "convikscale":
        scale = (hi - lo).clamp_min(1e-6).log().expand_as(convn) / 10.0
        return torch.cat([convn, xi, xk, scale], 1)
    raise ValueError(rep)


IN_CH = {"conv": 1, "conv4": 4, "convscale": 2, "ik": 2, "convik": 3, "convikscale": 4}
CHUNK = 20_000          # rows per h5 read; keeps the CPU-side transient at ~180 MB


class Bundle:
    """Every split resident on one device."""
    COLS = ("input16", "kernel9", "conv24", "label", "ii", "kk")

    def __init__(self, gen_dir=DATA, size=24, tnorm="minmax", device="cuda",
                 max_train=None, verbose=True, in_size=24):
        self.device = torch.device(device)
        # `size` is the TARGET resolution; `in_size` is the grid build_input assembles
        # on, always 24 (conv24's native support). Conflating them pads x and w to 48
        # while conv stays 24, and the cat in build_input fails.
        self.size, self.in_size, self.tnorm = size, in_size, tnorm
        nf = NORMS[tnorm]

        self.splits = {}
        for name in ("train", "val", "test"):
            cap = max_train if name == "train" else None
            with h5py.File(pathlib.Path(gen_dir) / f"gen_{name}.h5", "r", locking=False) as f:
                n = f["target48"].shape[0] if cap is None else min(cap, f["target48"].shape[0])
                d = {c: self._load(f[c], n) for c in self.COLS}
                t = torch.empty(n, size, size, device=self.device)
                for s0 in range(0, n, CHUNK):       # chunked: the last one must clip to n
                    e = min(s0 + CHUNK, n)
                    t[s0:e] = nf(pool_to(torch.from_numpy(f["target48"][s0:e]).to(self.device), size))
            d["t"] = t
            self.splits[name] = d
        if verbose:
            self._report()

    def _load(self, dset, n):
        """Straight to the device in chunks - never materialise a whole column on CPU."""
        out = torch.empty((n, *dset.shape[1:]),
                          dtype=torch.int32 if dset.dtype == np.int32 else torch.float32,
                          device=self.device)
        for s0 in range(0, n, CHUNK):
            e = min(s0 + CHUNK, n)
            out[s0:e] = torch.from_numpy(dset[s0:e]).to(self.device)
        return out

    def _report(self):
        for k in ("train", "val", "test"):
            v = self.splits[k]
            src, m = v["label"][:, 1], max(len(v["t"]), 1)
            mix = "/".join(f"{int((src == c).sum()) / m * 100:.0f}" for c in (1, 0, 2))
            print(f"  {k:6s} {v['t'].shape[0]:8d}  target {tuple(v['t'].shape[1:])}  "
                  f"[{v['t'].min():.3f}, {v['t'].max():.3f}]  C/E/N {mix}%  "
                  f"operands [{int(v['ii'].min())},{int(v['ii'].max())}]")
        held = sum(x.numel() * x.element_size()
                   for v in self.splits.values() for x in v.values())
        print(f"  GPU held: {held/1e9:.2f} GB")

    def batches(self, split, rep, batch_size=256, shuffle=False, limit=None,
                augment=False, gen=None):
        """Yield (x, t). augment applies the same dihedral transform to both."""
        s = self.splits[split]
        n = s["t"].shape[0] if limit is None else min(limit, s["t"].shape[0])
        order = torch.randperm(n, device=self.device, generator=gen) if shuffle \
            else torch.arange(n, device=self.device)
        for p in range(0, (n // batch_size) * batch_size, batch_size):
            b = order[p:p + batch_size]
            x = build_input(rep, s["input16"][b], s["kernel9"][b], s["conv24"][b], self.in_size)
            t = s["t"][b]
            if augment:
                k = int(torch.randint(0, 4, (1,), generator=gen, device=self.device).item())
                if k:
                    x, t = torch.rot90(x, k, (-2, -1)), torch.rot90(t, k, (-2, -1))
                if torch.rand(1, generator=gen, device=self.device).item() < 0.5:
                    x, t = torch.flip(x, (-1,)), torch.flip(t, (-1,))
            yield x, t

    def n_batches(self, split, batch_size=256, limit=None):
        n = self.splits[split]["t"].shape[0]
        return (min(limit, n) if limit is not None else n) // batch_size

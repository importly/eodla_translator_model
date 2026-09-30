"""Data assembly, entirely GPU-resident.

Each split is one self-contained file, data/gen_<split>.h5, whose rows hold the whole
sample. Only two things are derived at load, both per-sample and reversible: target48
is pooled to 24x24 and min-max normalised, and build_input min-max normalises conv24.
"""
import pathlib
from typing import Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from paths import DATA


def norm_minmax(t):
    lo = t.amin((-2, -1), keepdim=True)
    hi = t.amax((-2, -1), keepdim=True)
    return (t - lo) / (hi - lo + 1e-12)


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


def pad24(a):
    """Centre-pad a small map into the 24x24 conv grid, preserving alignment."""
    d = 24 - a.shape[-1]
    return F.pad(a, (d // 2, d - d // 2, d // 2, d - d // 2))


Rep = Literal["conv", "convikscale"]


def build_input(rep: Rep, x: torch.Tensor, w: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
    """Model input from one row's stored arrays. Nothing is looked up or recomputed.

    conv          the normalised conv - the task as originally specified
    convikscale   + x and w centre-padded to 24x24 + the conv's log dynamic range
    """
    conv = conv.unsqueeze(1)
    lo = conv.amin((-2, -1), keepdim=True)
    hi = conv.amax((-2, -1), keepdim=True)
    convn = (conv - lo) / (hi - lo + 1e-12)
    if rep == "conv":
        return convn
    if rep == "convikscale":
        scale = (hi - lo).clamp_min(1e-6).log().expand_as(convn) / 10.0
        return torch.cat([convn, pad24(x.unsqueeze(1)), pad24(w.unsqueeze(1)), scale], 1)
    raise ValueError(rep)


IN_CH = {"conv": 1, "convikscale": 4}


def signs(x, w):
    """Per-row signs (B, 1, 1) that flip x and w to their mostly-on side.
    output(-x, w) = output(x, -w) = -output(x, w) exactly, which min-max turns into
    1 - t, so the model only ever sees operands with most pixels on (README, "Operand
    signs"). An input with exactly 128 of 256 on is decided by its top-left pixel.
    """
    on_x = (x > 0).flatten(1).sum(1)
    sx = (on_x > 128) | ((on_x == 128) & (x[:, 0, 0] > 0))
    sw = (w > 0).flatten(1).sum(1) >= 41
    return (sx.float() * 2 - 1).view(-1, 1, 1), (sw.float() * 2 - 1).view(-1, 1, 1)


def predict(model, rep, x, w, conv):
    """Surrogate output (B, 24, 24) for any operands: flipped to their mostly-on side,
    then back."""
    sx, sw = signs(x, w)
    s = sx * sw
    y = model(build_input(rep, x * sx, w * sw, conv * s))[:, 0]
    return s * y + (1 - s) / 2


CHUNK = 20_000          # rows per h5 read; keeps the CPU-side transient at ~180 MB


class Bundle:
    """Every split resident on one device."""
    COLS = ("input16", "kernel9", "conv24", "label", "ii", "kk")

    def __init__(self, gen_dir=DATA, device="cuda", max_train=None, verbose=True):
        self.device = torch.device(device)
        self.splits = {}
        for name in ("train", "val", "test"):
            cap = max_train if name == "train" else None
            with h5py.File(pathlib.Path(gen_dir) / f"gen_{name}.h5", "r", locking=False) as f:
                n = f["target48"].shape[0] if cap is None else min(cap, f["target48"].shape[0])
                d = {c: self._load(f[c], n) for c in self.COLS}
                t = torch.empty(n, 24, 24, device=self.device)
                for s0 in range(0, n, CHUNK):       # chunked: the last one must clip to n
                    e = min(s0 + CHUNK, n)
                    t[s0:e] = norm_minmax(pool_to(torch.from_numpy(f["target48"][s0:e]).to(self.device), 24))
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

    def batches(self, split, rep, batch_size=256, shuffle=False, limit=None, gen=None):
        """Yield (x, t), each row with its operands flipped to their mostly-on side and t
        to match (`signs`); `predict` flips back.
        """
        s = self.splits[split]
        n = s["t"].shape[0] if limit is None else min(limit, s["t"].shape[0])
        order = torch.randperm(n, device=self.device, generator=gen) if shuffle \
            else torch.arange(n, device=self.device)
        for p in range(0, (n // batch_size) * batch_size, batch_size):
            b = order[p:p + batch_size]
            sx, sw = signs(s["input16"][b], s["kernel9"][b])
            sg = sx * sw
            x = build_input(rep, s["input16"][b] * sx, s["kernel9"][b] * sw, s["conv24"][b] * sg)
            yield x, sg * s["t"][b] + (1 - sg) / 2

    def n_batches(self, split, batch_size=256, limit=None):
        n = self.splits[split]["t"].shape[0]
        return (min(limit, n) if limit is not None else n) // batch_size

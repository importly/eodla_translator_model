"""Train the final translator with a chosen config and write a self-contained
artifact: checkpoint, config, metrics, and prediction figures.

Usage:
    uv run python final.py --cfg out/best_cfg.json --epochs 60 --tag v1
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import Bundle
from train import run, FLOOR

from paths import ROOT, RUNS

INK = "#1f2933"
MUTED = "#7b8794"
BLUE = "#3b6ea5"
RUST = "#a5643b"


def style(ax):
    """Minimal axes styling shared by the figures."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e4e7eb", lw=0.8)
    ax.set_axisbelow(True)


@torch.no_grad()
def fig_predictions(model, bundle, rep, path, n=8, seed=0):
    """Grid of n random test samples: conv input, prediction, target, |error|."""
    model.eval()
    x, t = next(bundle.batches("test", rep, batch_size=512))
    y = model(x)[:, 0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(y.shape[0], generator=g)[:n]
    conv = x[:, 0]
    rows = [("conv (input)", conv), ("predicted", y), ("target", t),
            ("|error|", (y - t).abs())]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.9 * n, 1.9 * len(rows) + 0.6),
                             constrained_layout=True)
    for r, (name, ten) in enumerate(rows):
        for j, i in enumerate(idx):
            a = ten[i].float().cpu().numpy()
            kw = dict(cmap="gray", vmin=0, vmax=1) if r < 3 else dict(cmap="magma", vmin=0, vmax=0.3)
            axes[r, j].imshow(a, interpolation="nearest", **kw)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            for s in axes[r, j].spines.values():
                s.set_visible(False)
            if r == 0:
                mse = F.mse_loss(y[i], t[i]).item()
                axes[r, j].set_title(f"mse {mse:.4f}", fontsize=8, color=MUTED)
        axes[r, 0].set_ylabel(name, fontsize=10, color=INK)
    fig.suptitle("Test predictions", fontsize=12, color=INK)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_curve(hist, floor, base_mse, path):
    """Validation MSE per epoch on a log axis against the conv baseline and the noise floor."""
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    ep = np.arange(len(hist))
    ax.plot(ep, hist, color=BLUE, lw=2, label="validation MSE")
    ax.axhline(base_mse, color=RUST, lw=1.5, ls="--", label="conv baseline")
    ax.axhline(floor, color=MUTED, lw=1.5, ls=":", label="noise floor")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("MSE (log)", color=INK)
    ax.set_title("Validation MSE against the two bounds", color=INK, fontsize=12)
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    # direct labels at the right edge
    ax.annotate(f"{hist[-1]:.5f}", (ep[-1], hist[-1]), textcoords="offset points",
                xytext=(6, 0), fontsize=9, color=BLUE, va="center")
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def fig_persample(model, bundle, rep, path):
    """Histogram of per-sample test MSE for the model vs the conv baseline; returns both arrays."""
    model.eval()
    m, b = [], []
    for x, t in bundle.batches("test", rep, batch_size=512):
        y = model(x)[:, 0]
        m.append(((y - t) ** 2).mean((-2, -1)).cpu())
        b.append(((x[:, 0] - t) ** 2).mean((-2, -1)).cpu())
    m, b = torch.cat(m).numpy(), torch.cat(b).numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    bins = np.logspace(np.log10(max(m.min(), 1e-6)), np.log10(b.max()), 50)
    ax.hist(b, bins=bins, color=RUST, alpha=0.7, label="conv baseline")
    ax.hist(m, bins=bins, color=BLUE, alpha=0.7, label="model")
    ax.set_xscale("log")
    ax.set_xlabel("per-sample MSE (log)", color=INK)
    ax.set_ylabel("test samples", color=INK)
    ax.set_title(f"Per-sample error  (median model {np.median(m):.5f}, "
                 f"baseline {np.median(b):.5f})", color=INK, fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return m, b


@torch.no_grad()
def conv_baseline(bundle, rep):
    """Test MSE of handing the normalised conv back as the prediction (the no-model baseline)."""
    se = n = 0
    for x, t in bundle.batches("test", "conv", batch_size=512):
        se += F.mse_loss(x[:, 0], t, reduction="sum").item()
        n += t.numel()
    return se / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--tag", type=str, default="final")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(ROOT / args.cfg))
    if "cfg" in cfg:                      # accept a sweep result row directly
        mk = cfg.get("model_kw", {})
        cfg = dict(cfg["cfg"]); cfg["model_kw"] = mk
    if args.epochs:
        cfg["epochs"] = args.epochs
    cfg["seed"] = args.seed
    cfg.pop("limit", None)

    outdir = RUNS / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    cfg["ckpt"] = str(outdir / "model.pt")     # also puts model_curves.png here

    print("config:", json.dumps(cfg))
    b = Bundle(size=24, tnorm="minmax", max_train=args.max_train)
    rep = cfg.get("rep", "conv")

    t0 = time.time()
    res, model = run(b, cfg, log_every=2)
    res["wall_minutes"] = (time.time() - t0) / 60

    base = conv_baseline(b, rep)
    floor = FLOOR[(24, "minmax")]
    res["conv_baseline_test_mse"] = base
    res["improvement_over_baseline"] = base / res["test"]["mse"]

    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "metrics": res},
               outdir / "model.pt")
    json.dump(res, open(outdir / "metrics.json", "w"), indent=1, default=float)
    json.dump(cfg, open(outdir / "config.json", "w"), indent=1)

    fig_predictions(model, b, rep, outdir / "predictions.png")
    fig_curve(res["hist"], floor, base, outdir / "val_curve.png")
    m, bb = fig_persample(model, b, rep, outdir / "per_sample.png")

    print("\n" + "=" * 70)
    print(f"test MSE        {res['test']['mse']:.6f}   PSNR {res['test']['psnr']:.2f} dB")
    print(f"noise floor     {floor:.6f}   -> {res['test']['mse']/floor:.2f}x floor")
    print(f"conv baseline   {base:.6f}   -> {base/res['test']['mse']:.1f}x better than baseline")
    print(f"train MSE       {res['train']['mse']:.6f}   gap {res['gap_train_val']:.2f}x")
    print(f"params {res['params']/1e6:.2f}M   epochs {res['epochs_ran']}   "
          f"{res['wall_minutes']:.1f} min")
    print(f"-> {outdir}")

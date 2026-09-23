"""Training loop and metrics for the EODLA translator.

Reports MSE against the measured irreducible noise floor, so a run is judged by
how much of the available headroom it closed rather than by an absolute number
that depends on target size and normalisation.
"""
import hashlib
import json
import math
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build
from data import IN_CH
from paths import RUNS

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, RUST = "#3b6ea5", "#a5643b"        # val, train - same colours as final.py

FLOOR = {  # per-pixel noise variance, measured by reference/analysis/floor.py
         # 2026-09-23, event camera, pilot3 data on the cluster
    (12, "minmax"): 0.000003, (12, "zscore"): 0.000053, (12, "robust"): 0.000004,
    (24, "minmax"): 0.000010, (24, "zscore"): 0.000174, (24, "robust"): 0.000012,
    (48, "minmax"): 0.000031, (48, "zscore"): 0.000637, (48, "robust"): 0.000040,
}


def grad_loss(y, t):
    """L1 distance between the finite differences (dy, dx) of prediction and target:
    penalises soft or misplaced edges independently of the pixel-wise term.
    """
    dy = y[..., 1:, :] - y[..., :-1, :]
    dx = y[..., :, 1:] - y[..., :, :-1]
    ty = t[..., 1:, :] - t[..., :-1, :]
    tx = t[..., :, 1:] - t[..., :, :-1]
    return F.l1_loss(dy, ty) + F.l1_loss(dx, tx)


def make_loss(kind, w_grad=0.0):
    """Training objective from a name ('l1' | 'l2' | 'huber') plus w_grad * grad_loss.
    Note HuberLoss with delta=1 on [0,1] targets is just MSE - kept only for reference.
    """
    base = {"l1": F.l1_loss, "l2": F.mse_loss,
            "huber": lambda y, t: F.huber_loss(y, t, delta=0.1)}[kind]
    if w_grad <= 0:
        return base
    return lambda y, t: base(y, t) + w_grad * grad_loss(y, t)


@torch.no_grad()
def evaluate(model, bundle, split, rep, batch_size=512, limit=None, lossf=None):
    """Test-time metrics for a split, on the model's current weights: per-pixel
    {mse, mae, psnr} in the target's normalised units (compare mse to FLOOR), plus
    the training objective as 'loss' when lossf is given.
    """
    model.eval()
    se = n = 0
    ae = 0.0
    losses = []
    for x, t in bundle.batches(split, rep, batch_size=batch_size, limit=limit):
        y = model(x)[:, 0]
        se += F.mse_loss(y, t, reduction="sum").item()
        ae += F.l1_loss(y, t, reduction="sum").item()
        n += t.numel()
        if lossf:
            losses.append(lossf(y, t).item())
    mse = se / n
    out = {"mse": mse, "mae": ae / n, "psnr": 10 * math.log10(1.0 / max(mse, 1e-12))}
    if lossf:
        out["loss"] = sum(losses) / len(losses)
    return out


def plot_curves(train_loss, val_loss, val_mse, floor, path, title):
    """Per epoch: train vs val loss (same objective), and val MSE against the noise
    floor with the best epoch - the one restored - marked. Redrawn every epoch so a
    run can be watched while it trains.
    """
    ep = list(range(len(val_mse)))
    best = min(ep, key=val_mse.__getitem__)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    a1.plot(ep, train_loss, color=RUST, lw=2, marker="o", ms=5, markevery=[-1], label="train")
    a1.plot(ep, val_loss, color=BLUE, lw=2, marker="o", ms=5, markevery=[-1], label="val")
    a1.set_title("loss (training objective)", color=INK, fontsize=11)
    a2.plot(ep, val_mse, color=BLUE, lw=2, label="val MSE")
    a2.axhline(floor, color=MUTED, lw=1.5, ls=":", label="noise floor")
    a2.plot(best, val_mse[best], "o", color=BLUE, ms=7, mec="white", mew=1.5)
    a2.annotate(f"best {val_mse[best]:.6f}  ({val_mse[best] / floor:.1f}x floor)",
                (best, val_mse[best]), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=INK)
    a2.set_title("validation MSE", color=INK, fontsize=11)
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.set_xlabel("epoch", color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9, which="both")
        ax.grid(axis="y", which="both", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(bundle, cfg, verbose=True, log_every=1):
    """Train one config and return (result_row, model).

    cfg keys: model, model_kw, rep, loss, w_grad, lr, wd, batch, epochs, sched
    ('onecycle' | 'cosine'), amp, clip, augment, seed, limit, ckpt (optional path).
    AdamW + OneCycle (no early stop) or cosine (+patience), bf16 autocast with a
    GradScaler, grad clipping; the best-val checkpoint is restored, test and train
    metrics evaluated once, and the weights ALWAYS saved (cfg['ckpt'] or an
    auto-named file under runs/auto/), with <ckpt>_curves.png beside them, redrawn
    every epoch. The row is what pipeline.py stores per run.
    """
    dev = bundle.device
    rep = cfg.get("rep", "conv")
    torch.manual_seed(cfg.get("seed", 0))

    model = build(cfg["model"], IN_CH[rep], **cfg.get("model_kw", {})).to(dev)
    nparam = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-3),
                            weight_decay=cfg.get("wd", 1e-4))
    epochs = cfg.get("epochs", 30)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.get("lr", 2e-3),
        total_steps=epochs * bundle.n_batches("train", cfg.get("batch", 256),
                                              cfg.get("limit")),
        pct_start=0.15,
    ) if cfg.get("sched", "onecycle") == "onecycle" else \
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    lossf = make_loss(cfg.get("loss", "l1"), cfg.get("w_grad", 0.0))
    amp = cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # ALWAYS save the trained weights. Explicit cfg["ckpt"] wins; otherwise an
    # auto-named file under runs/auto/ so no trained model is ever thrown away.
    ck = cfg.get("ckpt")
    if not ck:
        key = json.dumps({k: v for k, v in cfg.items() if k != "ckpt"}, sort_keys=True, default=str)
        ck = str(RUNS / "auto" /
                 f"{cfg['model']}_{rep}_{hashlib.md5(key.encode()).hexdigest()[:8]}.pt")
        cfg["ckpt"] = ck
    pathlib.Path(ck).parent.mkdir(parents=True, exist_ok=True)
    curves = pathlib.Path(ck).with_name(pathlib.Path(ck).stem + "_curves.png")
    if verbose:
        print(f"    curves -> {curves}", flush=True)

    floor = FLOOR[(bundle.size, bundle.tnorm)]
    best = {"mse": float("inf")}
    best_state = None
    hist, train_loss, val_loss = [], [], []
    stale = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        tl, steps = torch.zeros((), device=dev), 0     # summed on device: no sync per step
        for x, t in bundle.batches("train", rep, batch_size=cfg.get("batch", 256),
                                   shuffle=True, limit=cfg.get("limit"),
                                   augment=cfg.get("augment", False)):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = lossf(model(x)[:, 0], t)
            scaler.scale(loss).backward()
            if cfg.get("clip", 1.0):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(opt)
            scaler.update()
            if cfg.get("sched", "onecycle") == "onecycle":
                sched.step()
            tl += loss.detach()
            steps += 1
        if cfg.get("sched", "onecycle") != "onecycle":
            sched.step()

        vm = evaluate(model, bundle, "val", rep, lossf=lossf)
        hist.append(vm["mse"])
        train_loss.append(tl.item() / steps)
        val_loss.append(vm["loss"])
        plot_curves(train_loss, val_loss, hist, floor, curves,
                    f"{pathlib.Path(ck).stem}   {cfg['model']} / {rep}   epoch {ep + 1}/{epochs}")
        if vm["mse"] < best["mse"] - 1e-9:
            best = vm
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
            flag = " *"
        else:
            stale += 1
            flag = ""
        if verbose and (ep % log_every == 0 or flag):
            print(f"    ep {ep:3d}  loss train {train_loss[-1]:.5f} val {val_loss[-1]:.5f}  "
                  f"val mse {vm['mse']:.6f}  psnr {vm['psnr']:5.2f}  "
                  f"x floor {vm['mse']/floor:6.1f}  lr {opt.param_groups[0]['lr']:.2e}{flag}",
                  flush=True)
        # OneCycle's anneal phase is where the gains land; never cut it short.
        # Only cosine (which has no built-in end behaviour) uses patience.
        if cfg.get("sched", "onecycle") != "onecycle" and stale >= cfg.get("patience", 12):
            if verbose:
                print(f"    early stop at {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ck)
    tm = evaluate(model, bundle, "test", rep)
    trm = evaluate(model, bundle, "train", rep, batch_size=512, limit=20000)

    out = {
        "cfg": {k: v for k, v in cfg.items() if k != "model_kw"},
        "model_kw": cfg.get("model_kw", {}),
        "params": nparam,
        "epochs_ran": len(hist),
        "minutes": (time.time() - t0) / 60,
        "floor": floor,
        "val": best, "test": tm, "train": trm,
        "test_over_floor": tm["mse"] / floor,
        "gap_train_val": best["mse"] / max(trm["mse"], 1e-12),
        "hist": hist, "train_loss": train_loss, "val_loss": val_loss,
        "ckpt": cfg["ckpt"], "curves": str(curves),
    }
    if verbose:
        print(f"    -> test mse {tm['mse']:.6f}  psnr {tm['psnr']:.2f}  "
              f"{tm['mse']/floor:.1f}x floor  |  train {trm['mse']:.6f}  "
              f"gap {out['gap_train_val']:.1f}x  |  {nparam/1e6:.2f}M  "
              f"{out['minutes']:.1f} min", flush=True)
    return out, model


def save(results, path):
    """Write a list of result rows as indented JSON (floats coerced)."""
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=float)

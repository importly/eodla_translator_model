import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import h5py, numpy as np, torch, torch.nn.functional as F
from eodla_sim import TorchEODLASim

from paths import DATA

H5 = str(DATA / "simulated-data-v2.h5")
K = 16
with h5py.File(H5, "r") as f:
    inp = torch.from_numpy(f["input"][:K].astype(np.float32)).cuda()
    ker = torch.from_numpy(f["kernel"][:K].astype(np.float32)).cuda()

def target(full, crop=144, size=24):
    """Centre crop -> area resize -> per-sample min-max. The training target.

    Lived on TorchEODLASim until 2026-09-21; the live path (generate.py) crops and
    stores 48x48 instead, so it is kept here with the scripts that still use it.
    """
    H, W = full.shape[-2:]
    r0, c0 = (H - crop) // 2, (W - crop) // 2
    p = full[..., r0:r0 + crop, c0:c0 + crop].unsqueeze(1)
    t = F.interpolate(p, size=(size, size), mode="area").squeeze(1)
    lo = t.amin((-2, -1), keepdim=True)
    hi = t.amax((-2, -1), keepdim=True)
    return (t - lo) / (hi - lo + 1e-12)


def corr(a, b):
    a = (a - a.mean((-2,-1), keepdim=True)).flatten(1)
    b = (b - b.mean((-2,-1), keepdim=True)).flatten(1)
    return ((a*b).sum(1) / (a.norm(dim=1)*b.norm(dim=1)+1e-12)).mean().item()

# noise-free, so any difference is precision. The stored frames were made with the
# original bench model (README, "Bench model"), so they are not a reference any more
print("=== precision: complex64 vs complex128, noise-free (target 24x24) ===")
ref = None
for dt in [torch.complex128, torch.complex64]:
    try:
        s = TorchEODLASim(dtype=dt)
        torch.cuda.synchronize(); t0=time.time()
        ta = target(s.simulate(inp[:4], ker[:4], noise=False).float())
        torch.cuda.synchronize(); el=(time.time()-t0)/4
        ref = ta if ref is None else ref
        print(f"  {str(dt):22s} corr(vs complex128) {corr(ta,ref):+.6f}  "
              f"max diff {(ta-ref).abs().max():.1e}  {el*1000:.0f} ms/sample")
        del s; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {dt}: {type(e).__name__}: {e}")

print("\n=== throughput vs batch size (complex64) ===")
s = TorchEODLASim()
for B in [4, 8, 12, 16]:
    torch.cuda.reset_peak_memory_stats()
    s.simulate(inp[:B], ker[:B])  # warm
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(3):
        s.simulate(inp[:B], ker[:B])
    torch.cuda.synchronize()
    per = (time.time()-t0)/(3*B)
    print(f"  B={B:3d}  {per*1000:5.1f} ms/sample  "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB  "
          f"-> 100k in {per*100000/60:.0f} min")

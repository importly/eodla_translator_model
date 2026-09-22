"""Stage 1: operand pools -> data/operands.npz

16x16 inputs (CIFAR / EMNIST / noise) and free 81-bit 9x9 kernels, deduplicated and
sliced into contiguous pools:
    train [0, n_train)   val [n_train, +N_VAL)   test [.., n)
plus struct_kernels, held-out kernel families only the struct split draws from - see
README. The first n_train_struct train kernels come from those families too, minus
HELD_OUT.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, EMNIST
from PIL import Image

from paths import DATA

DS = DATA / "torchvision"
SRC_EMNIST, SRC_CIFAR, SRC_NOISE = 0, 1, 2          # label column 1

# 2M samples from N operands reuses each one 2M/N times. 200k halves that vs 100k;
# the cost is that 90k CIFAR draws come from a 50k split, so photos repeat under
# different augmentation. Drop back to 100_000 for one photo per operand.
N           = 200_000
N_VAL       = 5_000
N_TEST      = 5_000
SEED        = 1
FRAC_CIFAR  = 0.45
FRAC_EMNIST = 0.45        # remainder is random patterns
N_STRUCT    = 2_000       # kernels per family in the struct split
STRUCT      = ("uniform", "density", "smooth", "grating")     # struct_family 0..3
FRAC_STRUCT = 0.20        # of train kernels, from the struct families - see README
HELD_OUT    = "smooth"    # the one family training never sees


class Transpose:
    def __call__(self, img):
        return img.transpose(Image.Transpose.TRANSPOSE)     # EMNIST is stored transposed


def binarize(x):
    return ((x - x.mean()) / x.std()).sign()                # -> {-1, +1}


def cifar_pipeline(train):
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    return T.Compose(aug + [T.Resize((16, 16)), T.ToTensor(),
                            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            T.Grayscale(1), binarize])


def emnist_pipeline(train):
    aug = [T.RandomRotation(10)] if train else []
    return T.Compose([Transpose()] + aug + [T.Resize((16, 16)), T.ToTensor(),
                                            T.Normalize((0.1307,), (0.3081,)),
                                            T.Grayscale(1), binarize])


def load_sources(train, download=True):
    cifar = CIFAR10(str(DS / "cifar10"), train=train, download=download,
                    transform=cifar_pipeline(train))
    em = EMNIST(str(DS / "emnist"), split="byclass", train=train, download=download,
                transform=emnist_pipeline(train))
    t = em.targets.numpy() if isinstance(em.targets, torch.Tensor) else np.asarray(em.targets)
    return cifar, em, np.where(t < 10)[0]               # digit indices only


def make_kernels(n, seed):
    """Free 81-bit 9x9. The OG upsampled a 6x6 pattern, leaving 36 of 81 cells
    independent - see README."""
    rng = np.random.default_rng(seed)
    W = rng.integers(0, 2, (n, 9, 9), dtype=np.int8)
    while True:                                        # at 2**81 this never loops
        _, keep = np.unique(W.reshape(len(W), -1), axis=0, return_index=True)
        if len(keep) == len(W):
            return W.astype(np.float32) * 2 - 1
        W = np.concatenate([W[np.sort(keep)],
                            rng.integers(0, 2, (len(W) - len(keep), 9, 9), dtype=np.int8)])


def make_struct_kernels(n, seed):
    """n distinct kernels from each structured family, as +-1. density: iid at
    p in {.1,.25,.75,.9}. smooth: sign of a random 2x2..4x4 grid, bilinear-upsampled.
    grating: sign of an oriented cosine - edges, bars and stripes."""
    rng = np.random.default_rng(seed)
    m = 4 * n                                          # oversample; dedup drops repeats
    dens = rng.random((m, 9, 9)) < rng.choice([0.1, 0.25, 0.75, 0.9], (m, 1, 1))
    g = rng.choice([2, 3, 4], m)
    smooth = np.empty((m, 9, 9), bool)
    for k in (2, 3, 4):
        z = torch.from_numpy(rng.standard_normal((int((g == k).sum()), 1, k, k)))
        smooth[g == k] = F.interpolate(z, size=(9, 9), mode="bilinear",
                                       align_corners=True)[:, 0].numpy() >= 0
    y, x = np.mgrid[-4:5, -4:5]
    th = rng.uniform(0, np.pi, (m, 1, 1))
    f = rng.uniform(0.05, 0.5, (m, 1, 1))              # cycles/px: one edge .. Nyquist
    ph = rng.uniform(0, 2 * np.pi, (m, 1, 1))
    grating = np.cos(2 * np.pi * f * (x * np.cos(th) + y * np.sin(th)) + ph) >= 0
    out = []
    for K in (dens, smooth, grating):
        _, keep = np.unique(K.reshape(m, -1), axis=0, return_index=True)
        out.append(K[np.sort(keep)[:n]])
    return np.concatenate(out).astype(np.float32) * 2 - 1


def make_noise(n, seed):
    """n distinct random +-1 patterns at scales {2,4,8,16}; k divides 16, so
    block-repeat is exact. Scale 2 has only 16 patterns, so repeats are redrawn."""
    rng = np.random.default_rng(seed)

    def draw(n):
        X = np.empty((n, 16, 16), np.float32)
        ks = rng.choice([2, 4, 8, 16], n)
        for k in (2, 4, 8, 16):
            m = ks == k
            if m.any():
                p = rng.integers(0, 2, (int(m.sum()), k, k)).astype(np.float32) * 2 - 1
                X[m] = np.repeat(np.repeat(p, 16 // k, 1), 16 // k, 2)
        return X

    X = draw(n)
    while True:
        _, keep = np.unique(X.reshape(n, -1), axis=0, return_index=True)
        if len(keep) == n:
            return X
        X = np.concatenate([X[np.sort(keep)], draw(n - len(keep))])


def make_inputs(n, train, seed, frac_cifar=0.45, frac_emnist=0.45):
    """Blocks of CIFAR, EMNIST, noise. label = [class, source, index]."""
    n_cif, n_em = int(round(n * frac_cifar)), int(round(n * frac_emnist))
    n_noise = n - n_cif - n_em
    rng = np.random.default_rng(seed)
    X = np.empty((n, 16, 16), np.float32)
    L = np.empty((n, 3), np.float32)

    if n_cif or n_em:
        cifar, em, digits = load_sources(train)
        # with replacement once we want more rows than the split holds; augmentation
        # gives a different crop/flip each draw and exact repeats are dropped below
        for j, i in enumerate(rng.choice(len(cifar), n_cif, replace=n_cif > len(cifar))):
            img, lab = cifar[int(i)]
            X[j], L[j] = img[0].numpy(), (lab, SRC_CIFAR, j)
        for j, i in enumerate(rng.choice(len(digits), n_em, replace=n_em > len(digits))):
            img, _ = em[int(digits[i])]
            X[n_cif + j], L[n_cif + j] = img[0].numpy(), (0, SRC_EMNIST, n_cif + j)
    if n_noise:
        X[n_cif + n_em:] = make_noise(n_noise, seed + 1)
        L[n_cif + n_em:] = [(0, SRC_NOISE, n_cif + n_em + j) for j in range(n_noise)]
    return X, L


X, L = make_inputs(N, train=True, seed=SEED,
                   frac_cifar=FRAC_CIFAR, frac_emnist=FRAC_EMNIST)
W = make_kernels(N, seed=SEED)

p = np.random.default_rng(SEED).permutation(N)      # sources are laid out in blocks
X, L, W = X[p], L[p], W[p]

_, keep = np.unique(X.reshape(N, -1), axis=0, return_index=True)   # two photos can
keep = np.sort(keep)                                               # binarise alike
X, L, W = X[keep], L[keep], W[keep]

n = len(X)
n_train = n - N_VAL - N_TEST
# the control is ordinary test kernels, so the struct split never touches training ones
SK = np.concatenate([W[n_train + N_VAL:][:N_STRUCT], make_struct_kernels(N_STRUCT, SEED + 2)])

# the first n_ts train kernels come from the struct families except HELD_OUT, skipping
# any the struct split or val/test already has
fams = [s for s in STRUCT[1:] if s != HELD_OUT]
per = int(FRAC_STRUCT * n_train) // len(fams)
TK = make_struct_kernels(2 * per, SEED + 3).reshape(3, 2 * per, 9, 9)
seen = {k.tobytes() for k in np.concatenate([SK, W[n_train:]])}
picked = []
for s in fams:
    K = [k for k in TK[STRUCT.index(s) - 1] if k.tobytes() not in seen][:per]
    seen.update(k.tobytes() for k in K)
    picked += K
n_ts = len(picked)
W[:n_ts] = np.stack(picked)

np.savez_compressed(DATA / "operands.npz", inputs=X, kernels=W, labels=L,
                    n_train=n_train, n_val=N_VAL, n_test=N_TEST, n_train_struct=n_ts,
                    struct_kernels=SK, struct_names=np.array(STRUCT),
                    struct_family=np.repeat(np.arange(len(STRUCT)), N_STRUCT))

src = L[:, 1].astype(int)
print(f"{n} operands   train [0,{n_train})  val [{n_train},{n_train+N_VAL})  "
      f"test [{n_train+N_VAL},{n})")
print(f"  CIFAR {(src==1).sum()}  EMNIST {(src==0).sum()}  noise {(src==2).sum()}")
print(f"  train kernels [0,{n_ts}) from {'/'.join(fams)}, {HELD_OUT} held out")

"""Architecture zoo for the EODLA translator.

All models map (B, C, 24, 24) -> (B, 1, 24, 24). C depends on the input representation
(see data.py). CONV_TRACK and FULL_TRACK at the bottom are what actually gets trained.

Two tracks:
  conv-only   linear | cnn | linconv    see only conv24 - the task as specified
  full        resunet | fullres | fno | fnounet    also see the input and the kernel
"""
import torch
import torch.nn as nn


def act(name="silu"):
    """Activation module by name: 'silu' | 'relu' | 'gelu'."""
    return {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[name]()


def norm(kind, ch):
    """Normalisation module by kind: 'bn' (BatchNorm2d) | 'gn' (GroupNorm, <=8 groups) | none."""
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(min(8, ch), ch)
    return nn.Identity()


class ResBlock(nn.Module):
    """Two 3x3 convs (optionally dilated) with norm + activation and an identity or 1x1
    skip: out = act(F(x) + skip(x)). The unit both ResUNet and FullResNet are built from.
    """
    def __init__(self, cin, cout, dilation=1, nk="bn", a="silu", dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            norm(nk, cout), act(a),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False),
            norm(nk, cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        self.out_act = act(a)

    def forward(self, x):
        return self.out_act(self.conv(x) + self.skip(x))


class Up(nn.Module):
    """Learned 2x upsample (PixelShuffle), ICNR-initialised to start as nearest."""

    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Conv2d(cin, cout * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        w = self.proj.weight
        sub = torch.zeros(w.shape[0] // 4, *w.shape[1:])
        nn.init.kaiming_normal_(sub, nonlinearity="relu")
        with torch.no_grad():
            w.copy_(sub.repeat_interleave(4, dim=0))
            self.proj.bias.zero_()

    def forward(self, x):
        return self.shuffle(self.proj(x))


class Linear(nn.Module):
    """One affine map, conv24 -> target. No nonlinearity anywhere.

    The bench is linear up to intensity detection and the Poisson draw, so this measures
    how much of the task is not linear: whatever it cannot reach is what the nonlinear
    models are actually buying.
    """

    def __init__(self, in_ch=1, size=24, head="linear"):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(in_ch * size * size, size * size)
        self.act_out = nn.Sigmoid() if head == "sigmoid" else nn.Identity()

    def forward(self, x):
        return self.act_out(self.fc(x.flatten(1)).view(-1, 1, self.size, self.size))


class CNN(nn.Module):
    """Plain conv stack - no residual connections, no pooling, no skips.

    The middle baseline: everything the resunet family adds beyond stacked convolutions
    has to pay for itself against this.
    """

    def __init__(self, in_ch=1, base=64, blocks=6, nk="bn", a="silu", dropout=0.0,
                 head="linear"):
        super().__init__()
        layers, c = [], in_ch
        for i in range(blocks):
            layers += [nn.Conv2d(c, base, 3, padding=1), norm(nk, base), act(a)]
            if dropout > 0 and i >= blocks - 2:
                layers.append(nn.Dropout2d(dropout))
            c = base
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(base, 1, 1)
        self.act_out = nn.Sigmoid() if head == "sigmoid" else nn.Identity()

    def forward(self, x):
        return self.act_out(self.head(self.body(x)))


class LinConv(nn.Module):
    """Global affine path + local conv path, summed.

    The linear branch fits whatever is a linear function of the whole conv; the conv
    branch learns a residual correction on top. `blend` starts at zero, so training
    begins as exactly `linear` and the conv path only earns its way in.
    """

    def __init__(self, in_ch=1, size=24, base=64, blocks=6, nk="bn", a="silu",
                 dropout=0.0, head="linear"):
        super().__init__()
        self.lin = Linear(in_ch, size, head="linear")
        self.cnn = CNN(in_ch, base, blocks, nk, a, dropout, head="linear")
        self.blend = nn.Parameter(torch.zeros(1))
        self.act_out = nn.Sigmoid() if head == "sigmoid" else nn.Identity()

    def forward(self, x):
        return self.act_out(self.lin(x) + self.blend * self.cnn(x))


class ResUNet(nn.Module):
    """24 -> 12 -> 6 encoder/decoder with skip concatenation."""

    def __init__(self, in_ch=1, base=64, nk="bn", a="silu", dropout=0.0, head="linear"):
        super().__init__()
        b1, b2, b3 = base, base * 2, base * 4
        self.stem = nn.Conv2d(in_ch, b1, 3, padding=1)
        self.enc1 = nn.Sequential(ResBlock(b1, b1, nk=nk, a=a), ResBlock(b1, b1, nk=nk, a=a))
        self.enc2 = nn.Sequential(ResBlock(b1, b2, nk=nk, a=a), ResBlock(b2, b2, nk=nk, a=a))
        self.bott = nn.Sequential(ResBlock(b2, b3, nk=nk, a=a, dropout=dropout),
                                  ResBlock(b3, b3, nk=nk, a=a, dropout=dropout))
        self.up2 = Up(b3, b2)
        self.dec2 = nn.Sequential(ResBlock(b2 * 2, b2, nk=nk, a=a), ResBlock(b2, b2, nk=nk, a=a))
        self.up1 = Up(b2, b1)
        self.dec1 = nn.Sequential(ResBlock(b1 * 2, b1, nk=nk, a=a), ResBlock(b1, b1, nk=nk, a=a))
        self.head = nn.Conv2d(b1, 1, 1)
        self.act_out = nn.Sigmoid() if head == "sigmoid" else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        s1 = self.enc1(self.stem(x))
        s2 = self.enc2(self.pool(s1))
        z = self.bott(self.pool(s2))
        d = self.dec2(torch.cat([self.up2(z), s2], 1))
        d = self.dec1(torch.cat([self.up1(d), s1], 1))
        return self.act_out(self.head(d))


class FullResNet(nn.Module):
    """No pooling: a dilated residual stack at full 24x24 resolution.

    At this image size a handful of 3x3 layers already sees the whole frame, so
    the hierarchy a U-Net buys may not be worth the detail it destroys.
    """

    def __init__(self, in_ch=1, base=96, blocks=8, nk="bn", a="silu", dropout=0.0,
                 dilations=(1, 2, 4, 8), head="linear"):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.body = nn.Sequential(*[
            ResBlock(base, base, dilation=dilations[i % len(dilations)], nk=nk, a=a,
                     dropout=dropout if i >= blocks - 2 else 0.0)
            for i in range(blocks)
        ])
        self.head = nn.Conv2d(base, 1, 1)
        self.act_out = nn.Sigmoid() if head == "sigmoid" else nn.Identity()

    def forward(self, x):
        return self.act_out(self.head(self.body(self.stem(x))))


class SpectralConv2d(nn.Module):
    """Truncated-mode complex linear map in the Fourier domain (FNO core)."""

    def __init__(self, cin, cout, m1, m2):
        super().__init__()
        self.m1, self.m2, self.cout = m1, m2, cout
        s = 1.0 / (cin * cout)
        # Stored as REAL tensors (..., 2) = (re, im): AMP's GradScaler.unscale_ and
        # grad clipping have no complex kernels. view_as_complex in forward is free.
        self.w1 = nn.Parameter(s * torch.randn(cin, cout, m1, m2, 2))
        self.w2 = nn.Parameter(s * torch.randn(cin, cout, m1, m2, 2))

    def forward(self, x):
        # cuFFT has no bf16 path: run the spectral step in fp32 regardless of autocast
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            B, C, H, W = x.shape
            xf = torch.fft.rfft2(x)
            out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            m1, m2 = min(self.m1, H // 2), min(self.m2, W // 2 + 1)
            w1 = torch.view_as_complex(self.w1[:, :, :m1, :m2].contiguous())
            w2 = torch.view_as_complex(self.w2[:, :, :m1, :m2].contiguous())
            out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], w1)
            out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], w2)
            return torch.fft.irfft2(out, s=(H, W))


class FNO(nn.Module):
    """Fourier Neural Operator.

    The physical forward model is a chain of multiplications alternating between
    the Fourier domain (free-space propagation, the convolution itself) and the
    spatial domain (the two diffuser phase screens). An FNO layer is exactly that
    pair, which makes it a natural inductive bias for this problem.
    """

    def __init__(self, in_ch=1, width=64, layers=4, modes=12, a="silu", head="linear",
                 nk="gn", feature_out=False):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes)
                                   for _ in range(layers)])
        self.loc = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.nrm = nn.ModuleList([norm(nk, width) for _ in range(layers)])
        self.a = act(a)
        self.feature_out = feature_out
        if feature_out:
            self.proj = nn.Identity()
            self.act_out = nn.Identity()
        else:
            self.proj = nn.Sequential(nn.Conv2d(width, width * 2, 1), act(a),
                                      nn.Conv2d(width * 2, 1, 1))
            self.act_out = nn.Sigmoid() if head == "sigmoid" else nn.Identity()

    def forward(self, x):
        x = self.lift(x)
        for sp, lo, nm in zip(self.spec, self.loc, self.nrm):
            x = self.a(nm(sp(x) + lo(x).float())) + x.float()
        return self.act_out(self.proj(x))


class FNOUNet(nn.Module):
    """FNO trunk for global/Fourier structure, U-Net for local detail."""

    def __init__(self, in_ch=1, width=64, fno_layers=3, modes=12, base=48,
                 nk="bn", a="silu", dropout=0.0, head="linear"):
        super().__init__()
        self.fno = FNO(in_ch=in_ch, width=width, layers=fno_layers, modes=modes, a=a,
                       feature_out=True)
        self.unet = ResUNet(in_ch=width + in_ch, base=base, nk=nk, a=a, dropout=dropout,
                            head=head)

    def forward(self, x):
        return self.unet(torch.cat([self.fno(x), x], 1))


def build(name, in_ch, **kw):
    """Construct a model by name with keyword overrides; None values are dropped and only
    keys the architecture accepts are passed.
    Names: linear | cnn | linconv | resunet | fullres | fno | fnounet.
    """
    kw = {k: v for k, v in kw.items() if v is not None}
    keys = {
        "linear": {"size", "head"},
        "cnn": {"base", "blocks", "nk", "a", "dropout", "head"},
        "linconv": {"size", "base", "blocks", "nk", "a", "dropout", "head"},
        "resunet": {"base", "nk", "a", "dropout", "head"},
        "fullres": {"base", "blocks", "nk", "a", "dropout", "head"},
        "fno": {"width", "layers", "modes", "a", "head", "nk"},
        "fnounet": {"width", "fno_layers", "modes", "base", "nk", "a", "dropout", "head"},
    }
    cls = {"linear": Linear, "cnn": CNN, "linconv": LinConv,
           "resunet": ResUNet, "fullres": FullResNet, "fno": FNO, "fnounet": FNOUNet}
    if name not in cls:
        raise ValueError(f"unknown model {name}")
    return cls[name](in_ch=in_ch, **{k: v for k, v in kw.items() if k in keys[name]})


# ------------------------------------------------------------------- tracks
# What actually gets trained. Both predict the 24x24 pooled target and differ only in
# what the model is allowed to see. Entries are (name, cfg overrides).

# conv24 in, and nothing else - the model never sees the image or the kernel.
# Three capacity classes, so the gaps between them say how much of the task is linear,
# plus two reference points at the top of the range.
CONV_TRACK = [
    ("conv/linear",       dict(rep="conv",  model="linear",  model_kw=dict())),
    ("conv/cnn",          dict(rep="conv",  model="cnn",     model_kw=dict(base=64, blocks=6))),
    ("conv/linconv",      dict(rep="conv",  model="linconv", model_kw=dict(base=64, blocks=6))),
    ("conv/resunet-96",   dict(rep="conv",  model="resunet", model_kw=dict(base=96))),
    ("conv4/resunet-64",  dict(rep="conv4", model="resunet", model_kw=dict(base=64))),
]

# also sees the input image and the kernel, so it can beat the conv-only information
# limit. pipeline.py sweeps these as its arch stage.
FULL_TRACK = [
    ("resunet-48",    dict(model="resunet", model_kw=dict(base=48))),
    ("resunet-64",    dict(model="resunet", model_kw=dict(base=64))),
    ("resunet-96",    dict(model="resunet", model_kw=dict(base=96))),
    ("fullres-96x8",  dict(model="fullres", model_kw=dict(base=96, blocks=8))),
    ("fno-64x4",      dict(model="fno",     model_kw=dict(width=64, layers=4, modes=12))),
    ("fnounet-48",    dict(model="fnounet", model_kw=dict(width=48, fno_layers=3, base=48))),
]

# the reps that carry x and w; conv / convscale belong to CONV_TRACK
FULL_REPS = [
    ("ik", dict(rep="ik")),
    ("convik", dict(rep="convik")),
    ("convikscale", dict(rep="convikscale")),
]

"""Architectures for the EODLA translator.

All models map (B, C, 24, 24) -> (B, 1, 24, 24). C depends on the input representation
(see data.py).

  conv-only   linear | cnn | linconv    see only conv24 - the task as specified
  full        resunet | fullres         also see the input and the kernel
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Two 3x3 convs (optionally dilated) with BatchNorm + SiLU and an identity or 1x1
    skip: out = silu(F(x) + skip(x)). The unit both ResUNet and FullResNet are built from.
    """
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(),
            nn.Identity(),      # was dropout; keeps the checkpoint keys conv.4 / conv.5
            nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        self.out_act = nn.SiLU()

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

    The bench is linear up to intensity detection and the camera, so this measures how
    much of the task is not linear: whatever it cannot reach is what the nonlinear
    models are actually buying.
    """

    def __init__(self, in_ch=1, size=24):
        super().__init__()
        self.size = size
        self.fc = nn.Linear(in_ch * size * size, size * size)

    def forward(self, x):
        return self.fc(x.flatten(1)).view(-1, 1, self.size, self.size)


class CNN(nn.Module):
    """Plain conv stack - no residual connections, no pooling, no skips.

    The middle baseline: everything the resunet family adds beyond stacked convolutions
    has to pay for itself against this.
    """

    def __init__(self, in_ch=1, base=64, blocks=6):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(blocks):
            layers += [nn.Conv2d(c, base, 3, padding=1), nn.BatchNorm2d(base), nn.SiLU()]
            c = base
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        return self.head(self.body(x))


class LinConv(nn.Module):
    """Global affine path + local conv path, summed.

    The linear branch fits whatever is a linear function of the whole conv; the conv
    branch learns a residual correction on top. `blend` starts at zero, so training
    begins as exactly `linear` and the conv path only earns its way in.
    """

    def __init__(self, in_ch=1, size=24, base=64, blocks=6):
        super().__init__()
        self.lin = Linear(in_ch, size)
        self.cnn = CNN(in_ch, base, blocks)
        self.blend = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.lin(x) + self.blend * self.cnn(x)


class ResUNet(nn.Module):
    """24 -> 12 -> 6 encoder/decoder with skip concatenation."""

    def __init__(self, in_ch=1, base=64):
        super().__init__()
        b1, b2, b3 = base, base * 2, base * 4
        self.stem = nn.Conv2d(in_ch, b1, 3, padding=1)
        self.enc1 = nn.Sequential(ResBlock(b1, b1), ResBlock(b1, b1))
        self.enc2 = nn.Sequential(ResBlock(b1, b2), ResBlock(b2, b2))
        self.bott = nn.Sequential(ResBlock(b2, b3), ResBlock(b3, b3))
        self.up2 = Up(b3, b2)
        self.dec2 = nn.Sequential(ResBlock(b2 * 2, b2), ResBlock(b2, b2))
        self.up1 = Up(b2, b1)
        self.dec1 = nn.Sequential(ResBlock(b1 * 2, b1), ResBlock(b1, b1))
        self.head = nn.Conv2d(b1, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        s1 = self.enc1(self.stem(x))
        s2 = self.enc2(self.pool(s1))
        z = self.bott(self.pool(s2))
        d = self.dec2(torch.cat([self.up2(z), s2], 1))
        d = self.dec1(torch.cat([self.up1(d), s1], 1))
        return self.head(d)


class FullResNet(nn.Module):
    """No pooling: a dilated residual stack at full 24x24 resolution.

    At this image size a handful of 3x3 layers already sees the whole frame, so
    the hierarchy a U-Net buys may not be worth the detail it destroys.
    """

    def __init__(self, in_ch=1, base=96, blocks=8):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(base, base, dilation=(1, 2, 4, 8)[i % 4])
                                    for i in range(blocks)])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


MODELS = {"linear": Linear, "cnn": CNN, "linconv": LinConv,
          "resunet": ResUNet, "fullres": FullResNet}


def build(name, in_ch, **kw):
    """Construct a model by name, e.g. build("resunet", 4, base=96)."""
    return MODELS[name](in_ch=in_ch, **kw)


# The mentor's spec first: conv24 in, and nothing else. Three capacity classes, so the
# gaps between them say how much of the task is linear, then a strong net on the same
# input - where conv-only tops out. Last, that same net also shown x and w: the gap to
# it is what the conv alone cannot carry. Run by scripts/compare.py, one recipe.
TRACK = [
    ("conv/linear",            dict(rep="conv", model="linear",  model_kw=dict())),
    ("conv/cnn",               dict(rep="conv", model="cnn",     model_kw=dict(base=64, blocks=6))),
    ("conv/linconv",           dict(rep="conv", model="linconv", model_kw=dict(base=64, blocks=6))),
    ("conv/resunet-96",        dict(rep="conv", model="resunet", model_kw=dict(base=96))),
    ("convikscale/resunet-96", dict(rep="convikscale", model="resunet", model_kw=dict(base=96))),
]

import numpy as np
import torch
import torch.nn.functional as F

# event camera (DAVIS346), applied once to the combined output; light is in units of the
# all-on frame's peak. C: log-intensity step per event (the sensor's ~14% ON threshold).
# B: light level where the response turns from linear to log, set so 0728's digit strokes
# (0.8% of all-on) give ~4 events. BG_ON/OFF: background events per pixel per 16 ms
# window, measured on 0728. README, "Camera model".
C, B = 0.15, 0.01
BG_ON, BG_OFF = 0.0185, 0.0149


class TorchEODLASim:
    """GPU port of EODLASim. Build once (allocates the beam, the two diffuser phase
    screens seeded exactly like the original, and the three free-space transfer
    functions), then call simulate(inp16, ker9) for a batch of camera frames.

    Departures from the original, to match the bench (README, "Bench model" and "Camera
    model"): DMD patterns are binary blocks, not bilinear ramps; the negative pattern
    lights only the operand's footprint, not the whole frame; all four terms share one
    exposure; and the camera is an event camera applied once to the combined output,
    not Poisson photons on each capture.
    """
    def __init__(self, device="cuda", dtype=torch.complex64):
        self.device = torch.device(device)
        self.cdtype = dtype
        self.rdtype = torch.float32 if dtype == torch.complex64 else torch.float64

        N = self.N = 1080
        dx = self.dx = 7.6e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)

        z0, z1, z2, z3 = 0.53, 0.076 + 0.20, 0.08, 0.09
        lam = 514e-9
        fx = np.fft.fftfreq(N, dx)
        FX, FY = np.meshgrid(fx, fx)

        w0 = 0.9e-3
        zr = np.pi * w0**2 / lam
        r_z = z0 * (1 + (zr / z0) ** 4)
        w_z = w0 * np.sqrt(1 + (z0 / zr) ** 2)
        laser_power = 1.0

        # phase screens: seeded exactly as the original so the diffusers match
        np.random.seed(42)
        roughness = 0.55
        random_phase_0 = np.random.uniform(0, 2 * np.pi, (N, N)) * roughness
        random_phase_1 = np.random.uniform(0, 2 * np.pi, (N, N)) * roughness

        arg = 1.0 - (lam * FX) ** 2 - (lam * FY) ** 2
        k = 2 * np.pi / lam
        kz = np.where(arg >= 0, k * np.sqrt(np.maximum(arg, 0)), 0)

        H_0 = np.exp(1j * kz * z1); H_0[arg < 0] = 0
        H_1 = np.exp(1j * kz * z2); H_1[arg < 0] = 0
        H_2 = np.exp(1j * kz * z3); H_2[arg < 0] = 0

        # fftshift(A) * H then ifftshift(...) == A * ifftshift(H); fold the shift in
        self.H0s = self._t(np.fft.ifftshift(H_0))
        self.H1 = self._t(H_1)
        self.H2s = self._t(np.fft.ifftshift(H_2))

        E_gauss = (
            laser_power * w0 / w_z
            * np.exp(-(X**2 + Y**2) / w_z**2)
            * np.exp(-1j * (k * z0 + k * (X**2 + Y**2) / (2 * r_z) - np.arctan(z0 / zr)))
        )
        self.E_gauss = self._t(E_gauss)
        self.phase0 = self._t(np.exp(1j * random_phase_0))
        self.phase1 = self._t(np.exp(1j * random_phase_1))

        # fixed exposure: the all-on frame peaks at 1
        on_i = self._prep(torch.ones(1, 16, 16, device=self.device), 300)[0]
        on_k = self._prep(torch.ones(1, 9, 9, device=self.device), 168)[0]
        ref = self.resample(self.finish(self.kernel_field(on_k.to(dtype)),
                                        torch.fft.fft2(on_i.to(dtype))))
        self.scale = 1 / ref.amax().item()

    def _t(self, a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.device, self.cdtype)

    # ---- optical path -------------------------------------------------
    def kernel_field(self, kimg):
        """Steps that depend only on the kernel image: (B,N,N) real -> spectrum."""
        E_in = self.E_gauss * kimg
        E_prop = torch.fft.ifft2(torch.fft.fft2(E_in) * self.H0s)
        E_diff = E_prop * self.phase0
        return torch.fft.fft2(E_diff) * self.H1

    def finish(self, Epf, FM):
        """Convolution with the image, second propagation, second diffuser."""
        E_conv = torch.fft.ifft2(Epf * FM)
        E_prop = torch.fft.ifft2(torch.fft.fft2(E_conv) * self.H2s)
        E_diff = E_prop * self.phase1
        return (E_diff.real**2 + E_diff.imag**2)

    def resample(self, intensity):
        """1080 grid -> 346 camera pixels, crop 260 rows."""
        # RegularGridInterpolator over linspace(-N/2,N/2,N) queried at linspace(...,346)
        # is exactly bilinear with align_corners=True
        d = F.interpolate(
            intensity.unsqueeze(1), size=(346, 346), mode="bilinear", align_corners=True
        ).squeeze(1)
        return d[..., (346 - 260) // 2:(346 + 260) // 2, :]

    def camera(self, light, gen=None, noise=True):
        """Signed light -> signed event counts: log response, each count off by at most
        one, plus background events. noise=False returns the noise-free response."""
        x = light * self.scale
        m = x.sign() * torch.log1p(x.abs() / B) / C
        if not noise:
            return m
        u = torch.rand(m.shape, generator=gen, device=m.device)
        return (torch.floor(m + u)
                + torch.poisson(torch.full_like(m, BG_ON), generator=gen)
                - torch.poisson(torch.full_like(m, BG_OFF), generator=gen))

    # ---- full sample --------------------------------------------------
    def _prep(self, a, size):
        """+-1 operand -> the two binary DMD patterns on the 1080 grid. Each operand
        pixel is a block of mirrors; pos lights the +1 blocks, neg the -1 blocks, and
        nothing outside the operand is lit."""
        up = F.interpolate(a.unsqueeze(1), size=(size, size), mode="nearest").squeeze(1)
        p = (self.N - size) // 2
        return (F.pad((up > 0).to(self.rdtype), (p, p, p, p)),
                F.pad((up < 0).to(self.rdtype), (p, p, p, p)))

    @torch.no_grad()
    def simulate(self, inp16, ker9, gen=None, noise=True):
        """inp16 (B,16,16), ker9 (B,9,9) -> output-full (B,260,346)."""
        pos_i, neg_i = self._prep(inp16, 300)
        pos_k, neg_k = self._prep(ker9, 168)

        FM = {"p": torch.fft.fft2(pos_i.to(self.cdtype)),
              "n": torch.fft.fft2(neg_i.to(self.cdtype))}
        EPF = {"p": self.kernel_field(pos_k.to(self.cdtype)),
               "n": self.kernel_field(neg_k.to(self.cdtype))}

        # +(pos,pos) -(pos,neg) -(neg,pos) +(neg,neg); the camera sees only the result
        total = None
        for si, fk, kk in [(1, "p", "p"), (-1, "p", "n"), (-1, "n", "p"), (1, "n", "n")]:
            lit = si * self.finish(EPF[kk], FM[fk])
            total = lit if total is None else total + lit
        return self.camera(self.resample(total), gen, noise)

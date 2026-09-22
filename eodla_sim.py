import numpy as np
import torch
import torch.nn.functional as F


class TorchEODLASim:
    """GPU port of EODLASim. Build once (allocates the beam, the two diffuser phase
    screens seeded exactly like the original, and the three free-space transfer
    functions), then call simulate(inp16, ker9) for a batch of camera frames.
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

    def camera(self, intensity, gen=None):
        """Poisson shot noise at 50-photon peak, resample 1080->346, crop 260 rows."""
        peak = intensity.amax((-2, -1), keepdim=True)
        lam = intensity / (peak + 1e-30) * 50.0
        photons = torch.poisson(lam, generator=gen)
        # RegularGridInterpolator over linspace(-N/2,N/2,N) queried at linspace(...,346)
        # is exactly bilinear with align_corners=True
        d = F.interpolate(
            photons.unsqueeze(1), size=(346, 346), mode="bilinear", align_corners=True
        ).squeeze(1)
        return d[..., (346 - 260) // 2:(346 + 260) // 2, :]

    # ---- full sample --------------------------------------------------
    def _prep(self, a, size):
        """upsample -> pad to 1080 -> relu -> min-max over the FULL grid.

        Order matters and follows the original exactly: padding happens before
        the clamp and the min-max, so the zero border is included in the range
        and ``neg = 1 - pos`` is 1 across the whole aperture, not 0.
        """
        up = F.interpolate(a.unsqueeze(1), size=(size, size), mode="bilinear",
                           align_corners=False).squeeze(1)
        p = (self.N - size) // 2
        up = F.pad(up, (p, p, p, p))
        up = up.clamp(min=0)
        lo = up.amin((-2, -1), keepdim=True)
        hi = up.amax((-2, -1), keepdim=True)
        pos = (up - lo) / (hi - lo + 1e-30)
        return pos, 1.0 - pos

    @torch.no_grad()
    def simulate(self, inp16, ker9, gen=None):
        """inp16 (B,16,16), ker9 (B,9,9) -> output-full (B,260,346)."""
        pos_i, neg_i = self._prep(inp16, 300)
        pos_k, neg_k = self._prep(ker9, 168)

        FM = {"p": torch.fft.fft2(pos_i.to(self.cdtype)),
              "n": torch.fft.fft2(neg_i.to(self.cdtype))}
        EPF = {"p": self.kernel_field(pos_k.to(self.cdtype)),
               "n": self.kernel_field(neg_k.to(self.cdtype))}

        # +(pos,pos) -(pos,neg) -(neg,pos) +(neg,neg)
        total = None
        for si, fk, kk in [(1, "p", "p"), (-1, "p", "n"), (-1, "n", "p"), (1, "n", "n")]:
            cam = self.camera(self.finish(EPF[kk], FM[fk]), gen)
            total = si * cam if total is None else total + si * cam
        return total

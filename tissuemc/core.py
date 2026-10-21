"""Core Monte Carlo engine.

Units: lengths in cm, coefficients in 1/cm (the usual convention in tissue optics).

The photon-packet scheme follows MCML (Wang, Jacques & Zheng, Comput. Methods
Programs Biomed. 47, 131–146, 1995):

1. A packet enters normally at the origin with weight 1 - R_sp (specular reflection).
2. It travels a step s = -ln(xi) / mu_t, with mu_t = mu_a + mu_s.
3. If the step crosses a boundary, the packet is moved to the boundary and is either
   internally reflected (probability = Fresnel R) or escapes and is scored as
   diffuse reflectance (top) or transmittance (bottom). The unused part of the step
   is kept, measured in mean free paths.
4. Otherwise it deposits the fraction mu_a / mu_t of its weight (absorption) and
   scatters with the Henyey–Greenstein phase function.
5. Packets with weight below a threshold play Russian roulette, which keeps the
   estimator unbiased while ending the simulation.

All packets are propagated at once with NumPy arrays instead of one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COS_ZERO = 1.0 - 1.0e-12   # cos(theta) considered as 1 (grazing / normal incidence)
COS_90 = 1.0e-6            # cos(theta) considered as 0


@dataclass(frozen=True)
class Medium:
    """Homogeneous slab of turbid material.

    Parameters
    ----------
    mua : absorption coefficient [1/cm]
    mus : scattering coefficient [1/cm]
    g : anisotropy factor, mean cosine of the scattering angle (-1 < g < 1)
    n : refractive index of the slab
    thickness : slab thickness [cm]; use np.inf for a semi-infinite medium
    n_above, n_below : refractive indices of the ambient media
    """

    mua: float
    mus: float
    g: float = 0.0
    n: float = 1.0
    thickness: float = np.inf
    n_above: float = 1.0
    n_below: float = 1.0

    def __post_init__(self):
        if self.mua < 0 or self.mus < 0 or self.mua + self.mus == 0:
            raise ValueError("mua and mus must be non-negative and not both zero")
        if not -1.0 < self.g < 1.0:
            raise ValueError("g must lie in (-1, 1)")
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")

    @property
    def mut(self) -> float:
        """Total interaction coefficient mu_t = mu_a + mu_s [1/cm]."""
        return self.mua + self.mus

    @property
    def albedo(self) -> float:
        """Single-scattering albedo a = mu_s / mu_t."""
        return self.mus / self.mut

    @property
    def mus_reduced(self) -> float:
        """Reduced scattering coefficient mu_s' = mu_s (1 - g) [1/cm]."""
        return self.mus * (1.0 - self.g)

    @property
    def penetration_depth(self) -> float:
        """Diffusion-theory effective penetration depth delta = 1/mu_eff [cm]."""
        mueff = np.sqrt(3.0 * self.mua * (self.mua + self.mus_reduced))
        return np.inf if mueff == 0 else 1.0 / mueff


@dataclass
class SimulationResult:
    """Scored quantities, all normalized per incident photon."""

    medium: Medium
    n_photons: int
    specular: float
    diffuse_reflectance: float
    transmittance: float
    absorbed: float
    r_edges: np.ndarray = field(repr=False)
    z_edges: np.ndarray = field(repr=False)
    rd_r: np.ndarray = field(repr=False)        # diffuse reflectance per unit area, Rd(r) [1/cm^2]
    absorption_z: np.ndarray = field(repr=False)  # absorbed fraction per unit depth, A(z) [1/cm]
    rd_err: float = 0.0
    td_err: float = 0.0

    @property
    def total_reflectance(self) -> float:
        return self.specular + self.diffuse_reflectance

    @property
    def r_centers(self) -> np.ndarray:
        return 0.5 * (self.r_edges[1:] + self.r_edges[:-1])

    @property
    def z_centers(self) -> np.ndarray:
        return 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

    @property
    def fluence_z(self) -> np.ndarray:
        """Fluence rate phi(z) per unit incident irradiance (dimensionless): A(z) / mu_a."""
        if self.medium.mua == 0:
            raise ValueError("fluence is undefined from absorption when mua = 0")
        return self.absorption_z / self.medium.mua

    def energy_balance(self) -> float:
        """Rsp + Rd + T + A; should equal 1 up to roulette noise."""
        return self.specular + self.diffuse_reflectance + self.transmittance + self.absorbed

    def summary(self) -> str:
        m = self.medium
        return (
            f"Medium: mua={m.mua:g} 1/cm, mus={m.mus:g} 1/cm, g={m.g:g}, n={m.n:g}, "
            f"d={m.thickness:g} cm\n"
            f"Photons:              {self.n_photons:,}\n"
            f"Specular reflectance: {self.specular:.5f}\n"
            f"Diffuse reflectance:  {self.diffuse_reflectance:.5f} ± {self.rd_err:.5f}\n"
            f"Transmittance:        {self.transmittance:.5f} ± {self.td_err:.5f}\n"
            f"Absorbed:             {self.absorbed:.5f}\n"
            f"Energy balance:       {self.energy_balance():.5f}"
        )


def fresnel_reflectance(n1: float, n2: float, cos_i: np.ndarray) -> np.ndarray:
    """Unpolarized Fresnel reflectance going from index n1 into n2.

    cos_i is the cosine of the angle of incidence (non-negative). Returns 1 under
    total internal reflection.
    """
    cos_i = np.abs(np.asarray(cos_i, dtype=float))
    if n1 == n2:
        return np.zeros_like(cos_i)
    sin_i = np.sqrt(np.clip(1.0 - cos_i**2, 0.0, None))
    sin_t = n1 / n2 * sin_i
    r = np.ones_like(cos_i)                      # total internal reflection by default
    ok = sin_t < 1.0
    ci, st = cos_i[ok], sin_t[ok]
    ct = np.sqrt(1.0 - st**2)
    rs = ((n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)) ** 2
    rp = ((n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)) ** 2
    r[ok] = 0.5 * (rs + rp)
    return r


def henyey_greenstein(g: float, xi: np.ndarray) -> np.ndarray:
    """Sample cos(theta) from the Henyey–Greenstein phase function by inversion."""
    xi = np.asarray(xi, dtype=float)
    if abs(g) < 1e-9:
        return 2.0 * xi - 1.0
    tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
    return np.clip((1.0 + g * g - tmp * tmp) / (2.0 * g), -1.0, 1.0)


def _scatter(ux, uy, uz, cost, phi):
    """Rotate direction cosines by polar angle acos(cost) and azimuth phi (MCML eq. 3.24)."""
    sint = np.sqrt(np.clip(1.0 - cost**2, 0.0, None))
    cosp, sinp = np.cos(phi), np.sin(phi)
    nux = np.empty_like(ux)
    nuy = np.empty_like(uy)
    nuz = np.empty_like(uz)

    vertical = np.abs(uz) > COS_ZERO
    v = vertical
    nux[v] = sint[v] * cosp[v]
    nuy[v] = sint[v] * sinp[v]
    nuz[v] = cost[v] * np.sign(uz[v])

    o = ~vertical
    temp = np.sqrt(1.0 - uz[o] ** 2)
    nux[o] = sint[o] * (ux[o] * uz[o] * cosp[o] - uy[o] * sinp[o]) / temp + ux[o] * cost[o]
    nuy[o] = sint[o] * (uy[o] * uz[o] * cosp[o] + ux[o] * sinp[o]) / temp + uy[o] * cost[o]
    nuz[o] = -sint[o] * cosp[o] * temp + uz[o] * cost[o]
    return nux, nuy, nuz


def run(
    medium: Medium,
    n_photons: int = 100_000,
    *,
    seed: int | None = None,
    r_max: float = 1.0,
    nr: int = 100,
    z_max: float | None = None,
    nz: int = 100,
    weight_threshold: float = 1e-4,
    roulette_chance: float = 0.1,
    max_iterations: int = 1_000_000,
    phase=None,
) -> SimulationResult:
    """Run the Monte Carlo simulation for a pencil beam at normal incidence.

    Parameters
    ----------
    medium : the slab to simulate
    n_photons : number of photon packets launched
    seed : seed for the random generator (reproducible runs)
    r_max, nr : radial grid for Rd(r) [cm]
    z_max, nz : depth grid for A(z) [cm]; z_max defaults to the slab thickness or,
        for a semi-infinite medium, to ten transport mean free paths
    weight_threshold, roulette_chance : Russian roulette parameters
    phase : phase function used for scattering. None (default) uses Henyey–Greenstein
        with the medium's g. Any object with a ``sample(xi) -> cos(theta)`` method can
        be passed, e.g. ``tissuemc.mie.MiePhaseFunction``; its ``g`` attribute should
        match ``medium.g`` so that reported quantities such as mu_s' stay consistent.
    """
    if phase is not None and hasattr(phase, "g") and abs(phase.g - medium.g) > 1e-3:
        raise ValueError(
            f"phase function has g = {phase.g:.4f} but medium.g = {medium.g:.4f}; "
            "build the Medium with g=phase.g"
        )
    rng = np.random.default_rng(seed)
    m = medium
    mut = m.mut
    d = m.thickness

    if z_max is None:
        z_max = d if np.isfinite(d) else 10.0 / (m.mua + m.mus_reduced)
    r_edges = np.linspace(0.0, r_max, nr + 1)
    z_edges = np.linspace(0.0, z_max, nz + 1)

    # Specular reflection at the entrance surface (normal incidence)
    rsp = ((m.n_above - m.n) / (m.n_above + m.n)) ** 2

    n = int(n_photons)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    ux = np.zeros(n)
    uy = np.zeros(n)
    uz = np.ones(n)
    w = np.full(n, 1.0 - rsp)
    s_left = np.zeros(n)          # unused step, in mean free paths

    rd_hist = np.zeros(nr)
    a_hist = np.zeros(nz)
    rd_photon = np.zeros(n)       # per-photon scores, used for the statistical error
    td_photon = np.zeros(n)
    absorbed = 0.0

    alive = np.arange(n)
    it = 0
    while alive.size and it < max_iterations:
        it += 1
        i = alive

        # --- Step size -------------------------------------------------------
        need = s_left[i] == 0.0
        s_left[i[need]] = -np.log(rng.random(need.sum()) + 1e-300)
        s = s_left[i] / mut                       # geometric step [cm]

        # --- Distance to boundary ----------------------------------------------
        uzi = uz[i]
        db = np.full(i.size, np.inf)
        down = uzi > 0
        upw = uzi < 0
        if np.isfinite(d):
            db[down] = (d - z[i[down]]) / uzi[down]
        db[upw] = -z[i[upw]] / uzi[upw]

        hit = s > db
        hi, mi = i[hit], i[~hit]

        # --- Packets that reach a boundary -------------------------------------
        if hi.size:
            step = db[hit]
            x[hi] += ux[hi] * step
            y[hi] += uy[hi] * step
            z[hi] += uz[hi] * step
            s_left[hi] -= step * mut
            s_left[hi] = np.maximum(s_left[hi], 0.0)

            going_up = uz[hi] < 0
            n_out = np.where(going_up, m.n_above, m.n_below)
            cos_i = np.abs(uz[hi])
            R = np.empty(hi.size)
            for n2 in np.unique(n_out):
                sel = n_out == n2
                R[sel] = fresnel_reflectance(m.n, n2, cos_i[sel])
            reflect = rng.random(hi.size) <= R

            # internal reflection: mirror the direction, keep travelling
            refl = hi[reflect]
            uz[refl] = -uz[refl]

            # escape: score and kill
            esc = hi[~reflect]
            up_esc = esc[uz[esc] < 0]
            dn_esc = esc[uz[esc] > 0]
            if up_esc.size:
                r = np.hypot(x[up_esc], y[up_esc])
                rd_photon[up_esc] += w[up_esc]
                idx = np.searchsorted(r_edges, r, side="right") - 1
                ok = (idx >= 0) & (idx < nr)
                np.add.at(rd_hist, idx[ok], w[up_esc][ok])
            if dn_esc.size:
                td_photon[dn_esc] += w[dn_esc]
            w[esc] = 0.0

        # --- Packets that interact inside the medium ---------------------------
        if mi.size:
            step = s[~hit]
            x[mi] += ux[mi] * step
            y[mi] += uy[mi] * step
            z[mi] += uz[mi] * step
            s_left[mi] = 0.0

            dw = w[mi] * (m.mua / mut)
            w[mi] -= dw
            absorbed += dw.sum()
            idz = np.searchsorted(z_edges, z[mi], side="right") - 1
            ok = (idz >= 0) & (idz < nz)
            np.add.at(a_hist, idz[ok], dw[ok])

            xi = rng.random(mi.size)
            cost = henyey_greenstein(m.g, xi) if phase is None else phase.sample(xi)
            phi = 2.0 * np.pi * rng.random(mi.size)
            ux[mi], uy[mi], uz[mi] = _scatter(ux[mi], uy[mi], uz[mi], cost, phi)

            # Russian roulette
            low = mi[(w[mi] < weight_threshold) & (w[mi] > 0)]
            if low.size:
                survive = rng.random(low.size) < roulette_chance
                w[low[survive]] /= roulette_chance
                w[low[~survive]] = 0.0

        alive = alive[w[alive] > 0]

    if alive.size:
        raise RuntimeError("max_iterations reached with photons still alive")

    # Normalization per incident photon
    ring_area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)
    dz = np.diff(z_edges)

    rd = rd_photon.mean()
    td = td_photon.mean()
    return SimulationResult(
        medium=m,
        n_photons=n,
        specular=rsp,
        diffuse_reflectance=rd,
        transmittance=td,
        absorbed=absorbed / n,
        r_edges=r_edges,
        z_edges=z_edges,
        rd_r=rd_hist / (n * ring_area),
        absorption_z=a_hist / (n * dz),
        rd_err=rd_photon.std(ddof=1) / np.sqrt(n),
        td_err=td_photon.std(ddof=1) / np.sqrt(n),
    )

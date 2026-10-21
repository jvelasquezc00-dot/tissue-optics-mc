"""Mie theory for a homogeneous sphere and a sampleable Mie phase function.

Follows Bohren & Huffman, *Absorption and Scattering of Light by Small Particles*
(Wiley, 1983), appendix A (BHMIE), with the logarithmic derivative D_n(mx)
computed by downward recurrence for numerical stability.

Conventions
-----------
m : relative refractive index n_particle / n_medium (complex, Im(m) >= 0 for absorption)
x : size parameter 2*pi*a*n_medium / lambda_0, with a the sphere radius
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _n_terms(x: float) -> int:
    """Wiscombe's criterion for the number of terms in the Mie series."""
    return int(np.round(x + 4.0 * x ** (1.0 / 3.0) + 2.0))


def mie_coefficients(m: complex, x: float) -> tuple[np.ndarray, np.ndarray]:
    """External Mie coefficients a_n, b_n for n = 1..N."""
    if x <= 0:
        raise ValueError("size parameter x must be positive")
    m = complex(m)
    nmax = _n_terms(x)
    mx = m * x
    nmx = int(max(nmax, abs(mx))) + 16

    # Logarithmic derivative D_n(mx), downward recurrence
    d = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        d[n - 1] = n / mx - 1.0 / (d[n] + n / mx)

    # Riccati–Bessel functions psi_n(x) and xi_n(x) = psi_n - i chi_n, upward recurrence
    psi = np.zeros(nmax + 1)
    chi = np.zeros(nmax + 1)
    psi_m1, psi[0] = np.cos(x), np.sin(x)
    chi_m1, chi[0] = -np.sin(x), np.cos(x)
    psi[1] = psi[0] / x - psi_m1
    chi[1] = chi[0] / x - chi_m1
    for n in range(2, nmax + 1):
        psi[n] = (2 * n - 1) / x * psi[n - 1] - psi[n - 2]
        chi[n] = (2 * n - 1) / x * chi[n - 1] - chi[n - 2]
    xi = psi - 1j * chi

    n = np.arange(1, nmax + 1)
    dn = d[1 : nmax + 1]
    ta = dn / m + n / x
    tb = m * dn + n / x
    a = (ta * psi[n] - psi[n - 1]) / (ta * xi[n] - xi[n - 1])
    b = (tb * psi[n] - psi[n - 1]) / (tb * xi[n] - xi[n - 1])
    return a, b


def mie_efficiencies(m: complex, x: float) -> tuple[float, float, float]:
    """Extinction and scattering efficiencies Q_ext, Q_sca and asymmetry parameter g."""
    a, b = mie_coefficients(m, x)
    n = np.arange(1, a.size + 1)
    qext = 2.0 / x**2 * np.sum((2 * n + 1) * (a + b).real)
    qsca = 2.0 / x**2 * np.sum((2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2))
    nn = n[:-1]
    term1 = nn * (nn + 2) / (nn + 1) * (a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])).real
    term2 = (2 * n + 1) / (n * (n + 1)) * (a * np.conj(b)).real
    g = 4.0 / (x**2 * qsca) * (term1.sum() + term2.sum())
    return float(qext), float(qsca), float(g)


def mie_amplitudes(m: complex, x: float, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scattering amplitudes S1(mu), S2(mu) with mu = cos(theta)."""
    a, b = mie_coefficients(m, x)
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    nmax = a.size
    s1 = np.zeros(mu.size, dtype=complex)
    s2 = np.zeros(mu.size, dtype=complex)
    pi_prev = np.zeros_like(mu)   # pi_0
    pi_n = np.ones_like(mu)       # pi_1
    for n in range(1, nmax + 1):
        tau_n = n * mu * pi_n - (n + 1) * pi_prev
        f = (2 * n + 1) / (n * (n + 1))
        s1 += f * (a[n - 1] * pi_n + b[n - 1] * tau_n)
        s2 += f * (a[n - 1] * tau_n + b[n - 1] * pi_n)
        pi_next = ((2 * n + 1) * mu * pi_n - (n + 1) * pi_prev) / n
        pi_prev, pi_n = pi_n, pi_next
    return s1, s2


@dataclass
class MiePhaseFunction:
    """Unpolarized Mie phase function of a sphere, tabulated for fast sampling.

    p(mu) is normalized so that its integral over mu in [-1, 1] equals 1.
    Sampling inverts the cumulative distribution on a dense grid in theta,
    which resolves the sharp forward peak of large particles.
    """

    m: complex
    x: float
    n_angles: int = 4000

    def __post_init__(self):
        theta = np.linspace(0.0, np.pi, self.n_angles)
        mu = np.cos(theta)
        s1, s2 = mie_amplitudes(self.m, self.x, mu)
        intensity = 0.5 * (np.abs(s1) ** 2 + np.abs(s2) ** 2)
        # integrate over mu: dmu = sin(theta) dtheta
        w = intensity * np.sin(theta)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(theta))])
        norm = cdf[-1]
        self.theta = theta
        self.mu = mu
        self.p = intensity / norm
        self._cdf = cdf / norm
        self.qext, self.qsca, self.g = mie_efficiencies(self.m, self.x)

    def sample(self, xi: np.ndarray) -> np.ndarray:
        """Return cos(theta) samples for uniform random numbers xi in [0, 1)."""
        return np.cos(np.interp(xi, self._cdf, self.theta))

    def mean_cosine(self) -> float:
        """<cos theta> from the tabulated phase function (cross-check of g)."""
        f = self.mu * self.p * np.sin(self.theta)
        return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(self.theta)))


def size_parameter(diameter_um: float, wavelength_um: float, n_medium: float) -> float:
    """x = pi * d * n_medium / lambda_0."""
    return np.pi * diameter_um * n_medium / wavelength_um

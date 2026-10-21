"""Analytic references used to validate the Monte Carlo engine."""

from __future__ import annotations

import numpy as np


def chandrasekhar_h(albedo: float, mu: np.ndarray | float, n_quad: int = 200,
                    tol: float = 1e-12, max_iter: int = 10_000) -> np.ndarray:
    """Chandrasekhar H-function for isotropic scattering.

    Solved by fixed-point iteration of
        1/H(mu) = sqrt(1 - a) + (a/2) * integral_0^1 mu' H(mu') / (mu + mu') dmu'
    on a Gauss–Legendre grid.
    """
    x, wq = np.polynomial.legendre.leggauss(n_quad)
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * wq
    root = np.sqrt(1.0 - albedo)

    h = np.ones_like(nodes)
    for _ in range(max_iter):
        integral = np.sum(weights * nodes * h / (nodes[:, None] + nodes), axis=1)
        h_new = 1.0 / (root + 0.5 * albedo * integral)
        if np.max(np.abs(h_new - h)) < tol:
            h = h_new
            break
        h = h_new

    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    integral = np.sum(weights * nodes * h / (mu[:, None] + nodes), axis=1)
    return 1.0 / (root + 0.5 * albedo * integral)


def semi_infinite_isotropic_rd(albedo: float, mu0: float = 1.0) -> float:
    """Exact total diffuse reflectance of an index-matched semi-infinite medium with
    isotropic scattering (g = 0), for collimated incidence at cos(theta) = mu0:

        R(mu0) = 1 - H(mu0) * sqrt(1 - a)
    """
    return float(1.0 - chandrasekhar_h(albedo, mu0)[0] * np.sqrt(1.0 - albedo))


def internal_reflection_parameter(n: float) -> float:
    """A = (1 + r_i) / (1 - r_i), with the empirical fit of Groenhuis et al. (1983)
    for the internal diffuse reflectance r_i of a boundary with relative index n."""
    ri = -1.440 / n**2 + 0.710 / n + 0.668 + 0.0636 * n
    return (1.0 + ri) / (1.0 - ri)


def farrell_rd(r: np.ndarray, mua: float, mus: float, g: float, n: float = 1.0) -> np.ndarray:
    """Spatially resolved diffuse reflectance Rd(r) [1/cm^2] from diffusion theory
    with the dipole source model (Farrell, Patterson & Wilson, Med. Phys. 19, 879, 1992).

    Valid far from the source (r >> 1/mu_t') and for mua << mus'.
    """
    r = np.asarray(r, dtype=float)
    musp = mus * (1.0 - g)
    mutp = mua + musp
    a_red = musp / mutp
    mueff = np.sqrt(3.0 * mua * mutp)
    z0 = 1.0 / mutp
    zb = 2.0 * internal_reflection_parameter(n) / (3.0 * mutp)
    r1 = np.sqrt(z0**2 + r**2)
    r2 = np.sqrt((z0 + 2.0 * zb) ** 2 + r**2)
    term1 = z0 * (mueff + 1.0 / r1) * np.exp(-mueff * r1) / r1**2
    term2 = (z0 + 2.0 * zb) * (mueff + 1.0 / r2) * np.exp(-mueff * r2) / r2**2
    return a_red / (4.0 * np.pi) * (term1 + term2)

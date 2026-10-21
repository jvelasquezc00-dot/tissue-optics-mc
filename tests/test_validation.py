"""Validation of the Monte Carlo engine against independent references.

Tolerances are set from the statistical error of each run (several sigma), so the
tests check the physics, not luck.
"""

import numpy as np
import pytest

from tissuemc import Medium, fresnel_reflectance, henyey_greenstein, run
from tissuemc.analytic import farrell_rd, semi_infinite_isotropic_rd


def test_semi_infinite_isotropic_matches_chandrasekhar():
    """Index-matched semi-infinite medium, a = 0.9, g = 0: exact value from the H-function."""
    exact = semi_infinite_isotropic_rd(0.9)
    assert exact == pytest.approx(0.41495, abs=1e-4)
    res = run(Medium(mua=1.0, mus=9.0, g=0.0, n=1.0), 200_000, seed=11)
    assert abs(res.diffuse_reflectance - exact) < 4 * res.rd_err


def test_thin_slab_matches_van_de_hulst():
    """Slab with mua=10, mus=90 /cm, g=0.75, d=0.02 cm, n=1 (van de Hulst 1980; Prahl et al. 1989):
    Rd = 0.09739, Td = 0.66096."""
    res = run(Medium(mua=10.0, mus=90.0, g=0.75, n=1.0, thickness=0.02), 200_000, seed=12)
    assert abs(res.diffuse_reflectance - 0.09739) < 4 * res.rd_err
    assert abs(res.transmittance - 0.66096) < 4 * res.td_err


@pytest.mark.parametrize("n", [1.0, 1.37, 1.5])
def test_energy_is_conserved(n):
    res = run(Medium(mua=2.0, mus=50.0, g=0.9, n=n, thickness=0.1), 50_000, seed=13)
    assert res.energy_balance() == pytest.approx(1.0, abs=5e-3)


def test_non_absorbing_slab_has_no_absorption():
    res = run(Medium(mua=0.0, mus=50.0, g=0.8, thickness=0.05), 20_000, seed=14)
    assert res.absorbed == 0.0
    assert res.diffuse_reflectance + res.transmittance == pytest.approx(1.0, abs=1e-9)


def test_far_field_rd_agrees_with_diffusion_theory():
    """Far from the source the radial reflectance must follow the dipole model within ~15 %."""
    m = Medium(mua=0.1, mus=100.0, g=0.9, n=1.0)
    res = run(m, 200_000, seed=15, r_max=1.0, nr=50)
    r = res.r_centers
    window = (r > 0.2) & (r < 0.6)
    ratio = res.rd_r[window] / farrell_rd(r[window], m.mua, m.mus, m.g, m.n)
    assert np.median(ratio) == pytest.approx(1.0, abs=0.15)


def test_fresnel_limits():
    assert fresnel_reflectance(1.0, 1.5, np.array([1.0]))[0] == pytest.approx(0.04)
    # total internal reflection beyond the critical angle (~41.8 deg for n = 1.5)
    assert fresnel_reflectance(1.5, 1.0, np.array([np.cos(np.radians(50))]))[0] == 1.0
    assert np.all(fresnel_reflectance(1.33, 1.33, np.linspace(0, 1, 5)) == 0.0)


@pytest.mark.parametrize("g", [0.0, 0.5, 0.9, -0.3])
def test_henyey_greenstein_mean_cosine_equals_g(g):
    rng = np.random.default_rng(0)
    cost = henyey_greenstein(g, rng.random(400_000))
    assert cost.mean() == pytest.approx(g, abs=5e-3)


def test_medium_rejects_bad_parameters():
    with pytest.raises(ValueError):
        Medium(mua=1.0, mus=10.0, g=1.0)
    with pytest.raises(ValueError):
        Medium(mua=-1.0, mus=10.0)
    with pytest.raises(ValueError):
        Medium(mua=1.0, mus=10.0, thickness=0.0)

"""Validation of the Mie module."""

import numpy as np
import pytest

from tissuemc import Medium, MiePhaseFunction, run
from tissuemc.mie import mie_amplitudes, mie_efficiencies

# Reference values computed with miepython 3.3 (Prahl), an independent implementation.
# miepython writes the refractive index as n - ik, so 1.33 - 0.01j there equals 1.33 + 0.01j here.
REFERENCE = [
    # (m, x, Qext, Qsca, g)
    (1.5, 1.0, 0.21509759604288553, 0.21509759604288553, 0.19894249463608737),
    (1.33 + 0.01j, 10.0, 2.2492409080622853, 1.8721120605427188, 0.7541410676033555),
    (1.03, 6.75, 0.07916518684668983, 0.07916518684668983, 0.944488574977685),
    (1.55, 3.0, 3.7022013474601443, 3.7022013474601443, 0.7078636530707855),
]


@pytest.mark.parametrize("m, x, qext, qsca, g", REFERENCE)
def test_efficiencies_match_reference(m, x, qext, qsca, g):
    q = mie_efficiencies(m, x)
    assert q == pytest.approx((qext, qsca, g), rel=1e-9)


def test_rayleigh_limit():
    """Small spheres: Q_sca = (8/3) x^4 |(m^2-1)/(m^2+2)|^2 and g -> 0."""
    m, x = 1.5, 0.01
    qext, qsca, g = mie_efficiencies(m, x)
    k = (m**2 - 1) / (m**2 + 2)
    assert qsca == pytest.approx(8.0 / 3.0 * x**4 * abs(k) ** 2, rel=1e-3)
    assert abs(g) < 1e-3


def test_non_absorbing_sphere_has_qext_equal_qsca():
    qext, qsca, _ = mie_efficiencies(1.4, 25.0)
    assert qext == pytest.approx(qsca, rel=1e-10)


def test_forward_amplitude_gives_extinction():
    """Optical theorem: Q_ext = 4 Re{S(0)} / x^2."""
    m, x = 1.33 + 0.05j, 8.0
    s1, s2 = mie_amplitudes(m, x, np.array([1.0]))
    qext, _, _ = mie_efficiencies(m, x)
    assert s1[0] == pytest.approx(s2[0])
    assert 4.0 * s1[0].real / x**2 == pytest.approx(qext, rel=1e-10)


@pytest.mark.parametrize("m, x", [(1.03, 6.75), (1.5, 2.0), (1.2, 30.0)])
def test_phase_function_normalization_and_mean_cosine(m, x):
    pf = MiePhaseFunction(m, x)
    integrand = pf.p * np.sin(pf.theta)
    total = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(pf.theta))
    assert total == pytest.approx(1.0, abs=1e-6)
    assert pf.mean_cosine() == pytest.approx(pf.g, abs=2e-4)


def test_sampling_reproduces_g():
    pf = MiePhaseFunction(1.03, 6.75)
    rng = np.random.default_rng(1)
    assert pf.sample(rng.random(500_000)).mean() == pytest.approx(pf.g, abs=2e-3)


def test_run_rejects_inconsistent_g():
    pf = MiePhaseFunction(1.03, 6.75)
    with pytest.raises(ValueError):
        run(Medium(mua=1.0, mus=100.0, g=0.8), 10, phase=pf)


def test_mie_run_conserves_energy():
    pf = MiePhaseFunction(1.05, 5.0)
    res = run(Medium(mua=2.0, mus=60.0, g=pf.g, n=1.37, thickness=0.1), 20_000, seed=3, phase=pf)
    assert res.energy_balance() == pytest.approx(1.0, abs=5e-3)

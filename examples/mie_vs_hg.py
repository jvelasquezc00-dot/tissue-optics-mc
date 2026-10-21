"""Does the shape of the phase function matter? Mie vs Henyey–Greenstein with the same g.

Scatterers: spheres of 1 um diameter with relative index m = 1.03 (organelles in
cytoplasm) at 633 nm in a medium of index 1.36, which gives x = 6.75 and g = 0.944.
Both simulations use exactly the same mu_a, mu_s and g, so mu_s' is identical and
any difference comes only from the shape of the phase function.

Run from the repository root:
    python examples/mie_vs_hg.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tissuemc import Medium, MiePhaseFunction, run
from tissuemc.mie import size_parameter

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({
    "font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 150,
})


def phase_moment(pf, order):
    """<P_l(cos theta)> of a tabulated phase function."""
    leg = np.polynomial.legendre.Legendre.basis(order)(pf.mu)
    f = leg * pf.p * np.sin(pf.theta)
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(pf.theta)))


def main():
    x = size_parameter(diameter_um=1.0, wavelength_um=0.633, n_medium=1.36)
    pf = MiePhaseFunction(m=1.03, x=x)
    g = pf.g
    medium = Medium(mua=0.1, mus=100.0, g=g, n=1.0)

    g2_mie = phase_moment(pf, 2)
    gamma_mie = (1 - g2_mie) / (1 - g)
    gamma_hg = (1 - g**2) / (1 - g)   # for HG, g_2 = g^2

    hg = run(medium, 200_000, seed=21, r_max=0.5, nr=50)
    mie = run(medium, 200_000, seed=22, r_max=0.5, nr=50, phase=pf)
    r_mm = hg.r_centers * 10

    print(f"x = {x:.3f}, g = {g:.4f}, mu_s' = {medium.mus_reduced:.2f} 1/cm")
    print(f"gamma: Mie = {gamma_mie:.2f}, HG = {gamma_hg:.2f}")
    print(f"Rd total: HG = {hg.diffuse_reflectance:.4f} ± {hg.rd_err:.4f}, "
          f"Mie = {mie.diffuse_reflectance:.4f} ± {mie.rd_err:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    ax = axes[0]
    theta_deg = np.degrees(pf.theta)
    p_hg = (1 - g**2) / (2 * (1 + g**2 - 2 * g * pf.mu) ** 1.5)
    ax.semilogy(theta_deg, p_hg, color=ORANGE, lw=2, label=f"Henyey–Greenstein (γ = {gamma_hg:.2f})")
    ax.semilogy(theta_deg, pf.p, color=BLUE, lw=2, label=f"Mie (γ = {gamma_mie:.2f})")
    ax.set_xlabel("Scattering angle θ (deg)")
    ax.set_ylabel("Phase function p(cos θ)")
    ax.set_title(f"(a) Same g = {g:.3f}, different shape", loc="left", color=INK, fontsize=11)
    ax.set_xlim(0, 180)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogy(r_mm, hg.rd_r, color=ORANGE, lw=2, label="Henyey–Greenstein")
    ax.semilogy(r_mm, mie.rd_r, "o", color=BLUE, ms=4, mec="white", mew=0.7, label="Mie")
    ax.set_xlabel("Distance from the source, r (mm)")
    ax.set_ylabel(r"$R_d(r)$ (cm$^{-2}$)")
    ax.set_title("(b) Radial reflectance", loc="left", color=INK, fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.axvline(10 / medium.mus_reduced, color=MUTED, lw=1, ls=":")
    ax.text(10 / medium.mus_reduced + 0.08, 0.45, r"$1/\mu_s'$", color=MUTED, fontsize=9)
    ax.plot(r_mm, mie.rd_r / hg.rd_r, color=BLUE, lw=2)
    ax.set_xlabel("Distance from the source, r (mm)")
    ax.set_ylabel(r"$R_d^{\,Mie}(r)\,/\,R_d^{\,HG}(r)$")
    ax.set_title("(c) Ratio: the difference is near the source", loc="left", color=INK, fontsize=11)
    ax.set_ylim(0.3, 1.4)

    fig.suptitle(r"$\mu_a$ = 0.1 cm$^{-1}$, $\mu_s$ = 100 cm$^{-1}$, spheres d = 1 µm, m = 1.03, λ = 633 nm",
                 x=0.01, ha="left", color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "mie_vs_hg.png")
    plt.close(fig)
    print(f"Figure saved in {OUT / 'mie_vs_hg.png'}")


if __name__ == "__main__":
    main()

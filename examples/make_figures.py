"""Reproduce the figures shown in the README.

Run from the repository root:
    python examples/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tissuemc import Medium, run
from tissuemc.analytic import farrell_rd

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "figure.dpi": 150,
})


def radial_reflectance():
    """Monte Carlo Rd(r) against diffusion theory for a tissue-like medium."""
    m = Medium(mua=0.1, mus=100.0, g=0.9, n=1.0)
    res = run(m, 300_000, seed=1, r_max=1.0, nr=50)
    r = res.r_centers
    r_fine = np.linspace(0.01, 1.0, 300)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(r_fine, farrell_rd(r_fine, m.mua, m.mus, m.g, m.n), color=ORANGE,
                lw=2, label="Diffusion theory (Farrell et al. 1992)")
    ax.semilogy(r, res.rd_r, "o", color=BLUE, ms=4.5, mec="white", mew=0.8,
                label="Monte Carlo (this code)")
    ax.set_xlabel("Distance from the source, r (cm)")
    ax.set_ylabel(r"Diffuse reflectance $R_d(r)$ (cm$^{-2}$)")
    ax.set_title(r"$\mu_a$ = 0.1 cm$^{-1}$, $\mu_s$ = 100 cm$^{-1}$, g = 0.9",
                 color=INK, fontsize=11, loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "radial_reflectance.png")
    plt.close(fig)
    print(res.summary(), end="\n\n")


def fluence_vs_depth():
    """Fluence vs depth: absorption controls how deep light reaches."""
    shades = ["#86b6ef", "#2a78d6", "#184f95"]   # ordinal: light -> dark = weak -> strong absorption
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mua, color in zip([0.1, 1.0, 10.0], shades):
        m = Medium(mua=mua, mus=100.0, g=0.9, n=1.37)
        res = run(m, 200_000, seed=2, z_max=0.5, nz=100)
        ax.semilogy(res.z_centers * 10, res.fluence_z, color=color, lw=2,
                    label=rf"$\mu_a$ = {mua:g} cm$^{{-1}}$  ($\delta$ = {m.penetration_depth * 10:.2f} mm)")
    ax.set_xlabel("Depth, z (mm)")
    ax.set_ylabel(r"Fluence $\phi(z)$ / incident irradiance")
    ax.set_title(r"$\mu_s$ = 100 cm$^{-1}$, g = 0.9, n = 1.37 (tissue-like)",
                 color=INK, fontsize=11, loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fluence_vs_depth.png")
    plt.close(fig)


if __name__ == "__main__":
    radial_reflectance()
    fluence_vs_depth()
    print(f"Figures saved in {OUT}")

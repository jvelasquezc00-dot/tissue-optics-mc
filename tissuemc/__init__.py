"""tissuemc — Monte Carlo simulation of light transport in turbid media (tissue optics).

Vectorized NumPy implementation of the MCML algorithm (Wang, Jacques & Zheng, 1995)
for a single homogeneous slab, with Henyey–Greenstein phase function, Fresnel
boundaries and Russian roulette.
"""

from .core import Medium, SimulationResult, run, henyey_greenstein, fresnel_reflectance

__all__ = ["Medium", "SimulationResult", "run", "henyey_greenstein", "fresnel_reflectance"]
__version__ = "0.1.0"

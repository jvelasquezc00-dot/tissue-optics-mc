"""tissuemc — Monte Carlo simulation of light transport in turbid media (tissue optics).

Vectorized NumPy implementation of the MCML algorithm (Wang, Jacques & Zheng, 1995)
for a single homogeneous slab, with Henyey–Greenstein phase function, Fresnel
boundaries and Russian roulette.
"""

from .core import Medium, SimulationResult, run, henyey_greenstein, fresnel_reflectance
from .mie import MiePhaseFunction

__all__ = ["Medium", "SimulationResult", "run", "henyey_greenstein", "fresnel_reflectance",
           "MiePhaseFunction"]
__version__ = "0.2.0"

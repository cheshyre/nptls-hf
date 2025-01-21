# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import scipy.special
import numpy as np
import math
from nptlshf.core import State

hbarc = 197.3269788
Mproton = 938.2720  


def _compute_radial_wavefunction(hw, n, l, r):
    mass = Mproton
    bscale = (mass * hw) / (2 * hbarc * hbarc)
    norm = np.sqrt(
        np.sqrt((2 * bscale**3) / np.pi)
        * (np.power(2, n + 2 * l + 3) * math.factorial(n) * np.power(bscale, l))
        / (scipy.special.factorial2(2 * n + 2 * l + 1, exact=True))
    )
    laguerre = scipy.special.genlaguerre(n, l + 0.5)
    psirad = (
        norm * np.power(r, l) * np.exp(-bscale * r * r) * laguerre(2 * bscale * r * r)
    )
    return psirad * r


def compute_radial_wavefunction(state: State, hw: float, r: float):
    return _compute_radial_wavefunction(hw, state.n, state.l, r)

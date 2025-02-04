# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from nptlshf.core.hf_helper import HFHelper

system = "O16"
hw = 16
emax = 6

hf_helper = HFHelper()

hf_helper.set_emax(emax)
hf_helper.set_system(system)

occs = hf_helper.get_occupation_numbers()
density = np.diag(occs)

hf_helper.load_nn_hamiltonian(hw)
hf_helper.load_3n_hamiltonian(hw)

e = hf_helper.compute_energy(density)
f = hf_helper.compute_fock_matrix(density)

f_od = f - np.diag(np.diag(f))
f_od_norm = np.sum(np.power(f_od, 2))

print(f"E = {e}")
print(f"off-diagonal norm of F = {f_od_norm}")

# TODO: Write a code to iteratively diagonalize F and compute the HF energy


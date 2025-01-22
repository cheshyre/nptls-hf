# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

# from .check_2_2 import prob_2_2_ref_impl
from .check_2_1 import prob_2_1_ref_impl
from nptlshf.helpers import load_t, load_tcm_2b, load_vnn

def prob_2_3_ref_impl(basis, hw):
    '''
    Compute the intrinsic for He4 based on your basis and hw.

    Args:
        basis: List of HO State objects that is the single-particle basis.
        hw: Integer in [10, 12, ... , 40] MeV for the oscillator frequency.

    Returns:
        h1: 1-body part of Hamiltonian with kinetic energy.
        h2: 2-body part of Hamiltonian with V_NN and 2-body kinetic energy correction.
    '''
    dim = len(basis)

    h_1b = np.zeros((dim, dim))
    h_2b = np.zeros((dim, dim, dim, dim))

    # Your code here
    A = 4
    
    t = load_t(basis, hw)
    tcm_2b = load_tcm_2b(basis, hw)
    vnn = load_vnn(basis, hw)

    h_1b = (1 - 1/A) * t
    h_2b = vnn - 1/A * tcm_2b

    return h_1b, h_2b

def check_problem_2_3(get_intrinsic_hamiltonian_for_he4_from_basis_and_hw):

    for emax in [1, 2, 3]:
        basis = prob_2_1_ref_impl(emax)
        for hw in [16, 20]:
            h1_you, h2_you = get_intrinsic_hamiltonian_for_he4_from_basis_and_hw(basis, hw)
            h1_me, h2_me = prob_2_3_ref_impl(basis, hw)

            norm_diff_1b = np.sum(np.power(h1_you - h1_me, 2))
            norm_diff_2b = np.sum(np.power(h2_you - h2_me, 2))

            if norm_diff_1b > 1e-6:
                print("For emax = {}, hw = {}".format(emax, hw))
                print("1-body matrix elements do not match between our implementations.")
                return False
            if norm_diff_2b > 1e-6:
                print("For emax = {}, hw = {}".format(emax, hw))
                print("2-body matrix elements do not match between our implementations.")
                return False
    
    return True
            

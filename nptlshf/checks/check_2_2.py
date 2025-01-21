# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from .check_2_1 import prob_2_1_ref_impl

def prob_2_2_ref_impl(basis):
    occs = [0] * len(basis)
    for i, state in enumerate(basis):
        if state.n == 0 and state.l == 0:
            occs[i] = 1
    return occs


def check_problem_2_2(generate_he4_occupation_numbers_from_basis):

    for emax in range(2, 7):
        basis = prob_2_1_ref_impl(emax)

        occs = generate_he4_occupation_numbers_from_basis(basis)
        if len(occs) != len(basis):
            print(f"Length of occupations does not match length of basis.")
            return False

        for i, state in enumerate(basis):
            if state.n == 0 and state.l == 0:
                if occs[i] != 1:
                    print(f"Found occupation {occs[i]} for state {state} (expected: 1)")
                    return False
            else:
                if occs[i] != 0:
                    print(f"Found occupation {occs[i]} for state {state} (expected: 0)")
                    return False

    return True

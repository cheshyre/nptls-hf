# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from nptlshf.core import State

def prob_2_1_ref_impl(emax):
    states = []
    for e in range(emax + 1):
        for n in range(0, emax // 2 + 1):
            l = e - 2 * n
            for jj in [2 * l - 1, 2 * l + 1]:
                if jj < 0:
                    continue
                for mm in range(-1 * jj, jj + 1, 2):
                    for tt in [-1, 1]:
                        states.append(State(n, l, jj, mm, tt))
    
    return states

def check_problem_2_1(generate_emax_basis):

    for emax in range(14):
        your_basis = generate_emax_basis(emax)
        my_basis = prob_2_1_ref_impl(emax)

        if len(your_basis) != len(my_basis):
            print(f"For emax = {emax}, length of basis ({len(your_basis)}) does not match expectation ({len(my_basis)})")
            return False

        for state in my_basis:
            if state not in your_basis:
                print(f"For emax = {emax}, state {state} is missing from basis.")
                return False
            
    return True

# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from .basis import Basis, parse_system, make_occupations
from .mmh_file import MMHFile
from .sparse_operator import SparseOperator


class HFHelper:

    def __init__(self):
        self.emax = None
        self.basis = None
        
        self.system = None
        self.A = None
        self.occs = None

        self.ham_nn = None
        self.ham_3n = None

    
    def set_emax(self, emax):
        self.emax = emax
        self.basis = Basis.from_emax(self.emax)

        self.system = None
        self.A = None
        self.occs = None

        self.ham_nn = None
        self.ham_3n = None

    
    def set_system(self, system):
        if self.emax is None or self.basis is None:
            raise Exception("Please \"set_emax\" first before setting system.")
        
        self.system = system
        self.A, _, _ = parse_system(self.system)
        self.occs = make_occupations(self.system, self.basis)

        self.ham_nn = None
        self.ham_3n = None


    def get_states(self):
        return self.basis.states
    

    def get_occupation_numbers(self):
        return self.occs

    
    def load_nn_hamiltonian(self, hw):
        vnn_file = "data/emax_8/vnn_hw{}_e8_mscheme.mmh".format(hw)
        t_file = "data/emax_8/t_no_hw_e8_mscheme.mmh"
        tcm_file = "data/emax_8/tcm_no_hw_no_A_e8_mscheme.mmh"

        self.ham_nn= (
            SparseOperator(self.basis).read_from_file(MMHFile(vnn_file))
            + hw * SparseOperator(self.basis).read_from_file(MMHFile(t_file))
            - hw / self.A * SparseOperator(self.basis).read_from_file(MMHFile(tcm_file))
        )

    
    def load_3n_hamiltonian(self, hw):
        v3n_file = "data/emax_8/v3n_hw{}_e8_{}_mscheme.mmh".format(hw, self.system)
        self.ham_3n = SparseOperator(self.basis).read_from_file(MMHFile(v3n_file))

    
    def compute_energy(self, density):
        hamiltonian = self.ham_nn
        hamiltonian_3n = self.ham_3n

        e_1b = np.einsum("pq,pq", density, hamiltonian.one_body)
        e_2b = 0.0
        e_3b = 0.0
        for ich_pr, _ in enumerate(hamiltonian.basis.channels):
            for ich_qs, _ in enumerate(hamiltonian.basis.channels):
                for i_p, ppp in enumerate(hamiltonian.basis.states_in_channels[ich_pr]):
                    p, _ = ppp
                    for i_q, qqq in enumerate(hamiltonian.basis.states_in_channels[ich_qs]):
                        q, _ = qqq
                        for i_r, rrr in enumerate(hamiltonian.basis.states_in_channels[ich_pr]):
                            r, _ = rrr
                            for i_s, sss in enumerate(hamiltonian.basis.states_in_channels[ich_qs]):
                                s, _ = sss
                                e_2b += density[p, r] * density[q, s] * hamiltonian.two_body[ich_pr][ich_qs][i_p, i_q, i_r, i_s]
        if hamiltonian_3n is not None:
            for ich_pr, _ in enumerate(hamiltonian.basis.channels):
                for ich_qs, _ in enumerate(hamiltonian.basis.channels):
                    for i_p, ppp in enumerate(hamiltonian.basis.states_in_channels[ich_pr]):
                        p, _ = ppp
                        for i_q, qqq in enumerate(hamiltonian.basis.states_in_channels[ich_qs]):
                            q, _ = qqq
                            for i_r, rrr in enumerate(hamiltonian.basis.states_in_channels[ich_pr]):
                                r, _ = rrr
                                for i_s, sss in enumerate(hamiltonian.basis.states_in_channels[ich_qs]):
                                    s, _ = sss
                                    e_3b += density[p, r] * density[q, s] * hamiltonian_3n.two_body[ich_pr][ich_qs][i_p, i_q, i_r, i_s]
        return e_1b + 0.5 * e_2b + (1/6) * e_3b
    

    def compute_fock_matrix(self, density):
        hamiltonian = self.ham_nn
        hamiltonian_3n = self.ham_3n

        f_1b = hamiltonian.one_body
        f_2b = np.zeros_like(f_1b)
        f_3b = np.zeros_like(f_1b)
        for ich_pr, _ in enumerate(hamiltonian.basis.channels):
            for ich_qs, _ in enumerate(hamiltonian.basis.channels):
                for i_p, ppp in enumerate(hamiltonian.basis.states_in_channels[ich_pr]):
                    p, _ = ppp
                    for i_q, qqq in enumerate(hamiltonian.basis.states_in_channels[ich_qs]):
                        q, _ = qqq
                        for i_r, rrr in enumerate(hamiltonian.basis.states_in_channels[ich_pr]):
                            r, _ = rrr
                            for i_s, sss in enumerate(hamiltonian.basis.states_in_channels[ich_qs]):
                                s, _ = sss
                                f_2b[p, r] += density[q, s] * hamiltonian.two_body[ich_pr][ich_qs][i_p, i_q, i_r, i_s]
        if hamiltonian_3n is not None:
            for ich_pr, _ in enumerate(hamiltonian.basis.channels):
                for ich_qs, _ in enumerate(hamiltonian.basis.channels):
                    for i_p, ppp in enumerate(hamiltonian.basis.states_in_channels[ich_pr]):
                        p, _ = ppp
                        for i_q, qqq in enumerate(hamiltonian.basis.states_in_channels[ich_qs]):
                            q, _ = qqq
                            for i_r, rrr in enumerate(hamiltonian.basis.states_in_channels[ich_pr]):
                                r, _ = rrr
                                for i_s, sss in enumerate(hamiltonian.basis.states_in_channels[ich_qs]):
                                    s, _ = sss
                                    f_3b[p, r] += density[q, s] * hamiltonian_3n.two_body[ich_pr][ich_qs][i_p, i_q, i_r, i_s]
        return f_1b + f_2b + (1/2) * f_3b

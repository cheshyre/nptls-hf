# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np


from .basis import Basis
from .mmh_file import MMHFile


class SparseOperator:

    @classmethod
    def copy(cls, other: "SparseOperator"):
        new_op = cls(other.basis)

        new_op.zero_body = other.zero_body
        new_op.one_body = np.array(other.one_body)
        new_op.two_body = [
            [
                np.array(sub_mat)
                for sub_mat in row
            ]
            for row in other.two_body
        ]

        return new_op


    def __init__(self, basis: Basis, safe = True):
        self.basis = basis
        self.dim = len(basis.states)
        self.chan_dim = len(basis.channels)
        self.zero_body = 0.0
        self.one_body = np.zeros((self.dim, self.dim))
        self.two_body = [
            [
                np.zeros((len(states_pr), len(states_qs), len(states_pr), len(states_qs)))
                for states_qs in basis.states_in_channels
            ]
            for states_pr in basis.states_in_channels
        ]


    def read_from_file(self, file: MMHFile):
        self.zero_body = file.get_0b_part()

        for chan_pq, chan_pq_basis in zip(self.basis.channels, self.basis.states_in_channels):
            for p, ps in chan_pq_basis:
                for q, qs in chan_pq_basis:
                    self.one_body[p, q] = file.get_1b_matrix_element(ps, qs)

        for ich_pr, chan_pr in enumerate(self.basis.channels):
            chan_pr_basis = self.basis.states_in_channels[ich_pr]
            for ich_qs, chan_qs in enumerate(self.basis.channels):
                chan_qs_basis = self.basis.states_in_channels[ich_qs]
                for i_p, pp in enumerate(chan_pr_basis):
                    p, ps = pp
                    for i_q, qq in enumerate(chan_qs_basis):
                        q, qs = qq
                        for i_r, rr in enumerate(chan_pr_basis):
                            r, rs = rr
                            for i_s, ss in enumerate(chan_qs_basis):
                                s, ss = ss
                                self.two_body[ich_pr][ich_qs][i_p, i_q, i_r, i_s] = file.get_2b_matrix_element(ps, qs, rs, ss)
        
        return self


    def __iadd__(self, other: "SparseOperator"):
        if self.dim != other.dim:
            raise Exception(f"Operators have incompatible dimensions: {self.dim} vs {other.dim}.")
        
        self.zero_body += other.zero_body
        self.one_body += other.one_body
        for i in range(len(self.two_body)):
            for j in range(len(self.two_body[i])):
                self.two_body[i][j] += other.two_body[i][j]

        return self


    def __add__(self, other: "SparseOperator"):
        if self.dim != other.dim:
            raise Exception(f"Operators have incompatible dimensions: {self.dim} vs {other.dim}.")
        
        new_op = SparseOperator.copy(self)
        new_op += other

        return new_op


    def __isub__(self, other: "SparseOperator"):
        if self.dim != other.dim:
            raise Exception(f"Operators have incompatible dimensions: {self.dim} vs {other.dim}.")
        
        self.zero_body -= other.zero_body
        self.one_body -= other.one_body
        for i in range(len(self.two_body)):
            for j in range(len(self.two_body[i])):
                self.two_body[i][j] -= other.two_body[i][j]

        return self


    def __sub__(self, other: "SparseOperator"):
        if self.dim != other.dim:
            raise Exception(f"Operators have incompatible dimensions: {self.dim} vs {other.dim}.")
        
        new_op = SparseOperator.copy(self)
        new_op -= other

        return new_op


    def __imul__(self, factor: float):

        self.zero_body *= factor
        self.one_body *= factor
        for i in range(len(self.two_body)):
            for j in range(len(self.two_body[i])):
                self.two_body[i][j] *= factor

        return self


    def __mul__(self, factor: float):
        new_op = SparseOperator.copy(self)
        new_op *= factor

        return new_op


    def __rmul__(self, factor: float):
        new_op = SparseOperator.copy(self)
        new_op *= factor

        return new_op


    def __idiv__(self, factor: float):
        if factor == 0.0:
            raise Exception("Dividing by zero.")

        self.zero_body /= factor
        self.one_body /= factor
        for i in range(len(self.two_body)):
            for j in range(len(self.two_body[i])):
                self.two_body[i][j] /= factor

        return self


    def __div__(self, factor: float):
        if factor == 0.0:
            raise Exception("Dividing by zero.")

        new_op = SparseOperator.copy(self)
        new_op /= factor

        return new_op

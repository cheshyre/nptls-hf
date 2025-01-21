# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np


from .basis import Basis
from .mmh_file import MMHFile


class DenseOperator:

    @classmethod
    def copy(cls, other: "DenseOperator"):
        new_op = cls(other.basis)

        new_op.zero_body = other.zero_body
        new_op.one_body = np.array(other.one_body)
        new_op.two_body = np.array(other.two_body)

        return new_op


    def __init__(self, basis: Basis, safe = True):
        self.basis = basis
        self.dim = len(basis.states)
        if safe and self.dim > 150:
            memory_consumption = self.dim**4 * 8 / (2**30)
            raise Exception(f"Memory consumption with basis size of {self.dim} is {memory_consumption:.2f} GB. This is dangerous.")
        self.zero_body = 0.0
        self.one_body = np.zeros((self.dim, self.dim))
        self.two_body = np.zeros((self.dim, self.dim, self.dim, self.dim))


    def read_from_file(self, file: MMHFile):
        self.zero_body = file.get_0b_part()

        for chan_pq, chan_pq_basis in zip(self.basis.channels, self.basis.states_in_channels):
            for p, ps in chan_pq_basis:
                for q, qs in chan_pq_basis:
                    self.one_body[p, q] = file.get_1b_matrix_element(ps, qs)

        for chan_pr, chan_pr_basis in zip(self.basis.channels, self.basis.states_in_channels):
            for chan_qs, chan_qs_basis in zip(self.basis.channels, self.basis.states_in_channels):
                for p, ps in chan_pr_basis:
                    for q, qs in chan_qs_basis:
                        for r, rs in chan_pr_basis:
                            for s, ss in chan_qs_basis:
                                self.two_body[p, q, r, s] = file.get_2b_matrix_element(ps, qs, rs, ss)
                                self.two_body[q, p, r, s] = -1 * self.two_body[p, q, r, s]
                                self.two_body[p, q, s, r] = -1 * self.two_body[p, q, r, s]
                                self.two_body[q, p, s, r] = self.two_body[p, q, r, s]
        
        return self


    def __iadd__(self, other: "DenseOperator"):
        if self.dim != other.dim:
            raise Exception(f"Operators have incompatible dimensions: {self.dim} vs {other.dim}.")
        
        self.zero_body += other.zero_body
        self.one_body += other.one_body
        self.two_body += other.two_body

        return self


    def __add__(self, other: "DenseOperator"):
        if self.dim != other.dim:
            raise Exception(f"Operators have incompatible dimensions: {self.dim} vs {other.dim}.")
        
        new_op = DenseOperator.copy(self)
        new_op += other

        return new_op


    def __isub__(self, other: "DenseOperator"):
        if self.dim != other.dim:
            raise Exception(f"Operators have incompatible dimensions: {self.dim} vs {other.dim}.")
        
        self.zero_body -= other.zero_body
        self.one_body -= other.one_body
        self.two_body -= other.two_body

        return self


    def __sub__(self, other: "DenseOperator"):
        if self.dim != other.dim:
            raise Exception(f"Operators have incompatible dimensions: {self.dim} vs {other.dim}.")
        
        new_op = DenseOperator.copy(self)
        new_op -= other

        return new_op


    def __imul__(self, factor: float):

        self.zero_body *= factor
        self.one_body *= factor
        self.two_body *= factor

        return self


    def __mul__(self, factor: float):
        new_op = DenseOperator.copy(self)
        new_op *= factor

        return new_op


    def __rmul__(self, factor: float):
        new_op = DenseOperator.copy(self)
        new_op *= factor

        return new_op


    def __idiv__(self, factor: float):
        if factor == 0.0:
            raise Exception("Dividing by zero.")

        self.zero_body /= factor
        self.one_body /= factor
        self.two_body /= factor

        return self


    def __div__(self, factor: float):
        if factor == 0.0:
            raise Exception("Dividing by zero.")

        new_op = DenseOperator.copy(self)
        new_op /= factor

        return new_op

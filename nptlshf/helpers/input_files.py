# Copyright (c) 2025 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from pathlib import Path

from nptlshf.core.dense_operator import DenseOperator
from nptlshf.core.basis import Basis
from nptlshf.core.mmh_file import MMHFile

base_path = Path(__file__).absolute().parent.parent.parent


def load_vnn(basis, hw):
    vnn_file = base_path / f"data/emax_4/vnn_hw{hw}_e4_mscheme.mmh"

    return DenseOperator(Basis(basis)).read_from_file(MMHFile(vnn_file)).two_body


def load_v3n(basis, hw, system):
    system = system.title()
    v3n_file = base_path / f"data/emax_4/v3n_hw{hw}_e4_{system}_mscheme.mmh"

    return DenseOperator(Basis(basis)).read_from_file(MMHFile(v3n_file)).two_body


def load_t(basis, hw):
    t_file = base_path / "data/emax_4/t_no_hw_e4_mscheme.mmh"

    return hw * DenseOperator(Basis(basis)).read_from_file(MMHFile(t_file)).one_body


def load_tcm_2b(basis, hw):
    tcm_file = base_path / "data/emax_4/tcm_no_hw_no_A_e4_mscheme.mmh"

    return hw * DenseOperator(Basis(basis)).read_from_file(MMHFile(tcm_file)).two_body

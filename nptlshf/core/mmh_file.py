# Copyright (c) 2025 Matthias Heinz
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import re
import pathlib
import pickle

import numpy as np

from .chan import Channel
from .state import State


def _read_lines(fname, start_pattern, end_pattern=None, ignore_pattern=None):
    reading = False
    lines = []
    with open(fname) as fin:
        for line in fin:
            if not reading:
                if start_pattern in line:
                    reading = True
            else:
                if end_pattern is not None and end_pattern in line:
                    return lines
                else:
                    if ignore_pattern is None or ignore_pattern not in line:
                        lines.append(line.strip())
                # if (
                #     end_pattern is None or end_pattern not in line
                # ) and (
                #     ignore_pattern is None or ignore_pattern not in line
                # ):
                #     lines.append(line.strip())
                # else:
                #     return lines
    return lines


def _read_channels(fname):
    lines = _read_lines(fname, "i_ch    p   tt   jj |mm|", "States:")
    channels = []
    for line in lines:
        ich, p, tt, jj, amm = [int(x) for x in line.split()]

        channels.append((ich, Channel(p, jj, amm, tt)))

    return channels


def _read_basis(fname):
    lines = _read_lines(fname, " i_p    n    l   jj   tt   mm", "0B Part:")

    basis = []
    for line in lines:
        ip, n, l, jj, tt, mm = [int(x) for x in line.split()]

        basis.append((ip, State(n, l, jj, mm, tt)))

    return basis


def _read_2bchannels(fname):
    tbchans = []
    with open(fname) as fin:
        for line in fin:
            if "Chan_pqrs" in line:
                match = re.search(r"pr=(\d+),qs=(\d+)", line)
                tbchans.append((int(match.group(1)), int(match.group(2))))

    return tbchans


def _read_0b_part(fname):
    with open(fname) as fin:
        for line in fin:
            if "0B Part:" in line:
                return float(line.strip().split()[-1])
    return 0.0


def _read_1b_mes(fname):
    lines = _read_lines(fname, "1B Part:", "2B Part:")

    mes = {
        tuple([int(x) for x in line.split()[:-1]]): float(line.split()[-1])
        for line in lines
    }
    return mes


def _read_2b_mes(fname):
    lines = _read_lines(fname, "2B Part:", ignore_pattern="Chan_pqrs")

    mes = {
        tuple([int(x) for x in line.split()[:-1]]): float(line.split()[-1])
        for line in lines
    }
    return mes


class MMHFileStructure:
    def __init__(self, fname):
        self.basis = _read_basis(fname)
        self.channels = _read_channels(fname)
        self.chans_2b = _read_2bchannels(fname)

        # Convert universal sid and cid indices into file-specific indices
        self.sid_to_ip_map = {state.sid: ip for ip, state in self.basis}
        # max_sid = max([x for x in self.sid_to_ip_map])
        # self.sid_to_ip_dense_lookup = [-1] * (max_sid + 1)
        # for x in self.sid_to_ip_map:
        #     self.sid_to_ip_dense_lookup[x] = self.sid_to_ip_map[x]
        self.cid_to_ich_map = {chan.cid: ich for ich, chan in self.channels}

        # Lookups for ip -> ich, local_ip and ich -> ch_dim
        self.ch_dims = [0] * len(self.channels)
        self.state_chans = [-1] * len(self.basis)
        self.state_local_indices = [-1] * len(self.basis)
        for ip, state in self.basis:
            ich = self.cid_to_ich_map[state.chan.cid]
            self.state_local_indices[ip] = self.ch_dims[ich]
            self.state_chans[ip] = ich
            self.ch_dims[ich] += 1

        # Lookup for (ich_pr, ich_qs) -> local_2bch_index
        self.local_2bch_indices = {
            ch_tuple: i for i, ch_tuple in enumerate(self.chans_2b)
        }


class MMHFileData:
    def __init__(self, dim_1b, dims_2b):
        self.data_0b = 0
        self.data_1b = np.zeros(shape=(dim_1b, dim_1b))
        self.data_2b = [
            np.zeros(shape=(dim_pr, dim_qs, dim_pr, dim_qs))
            for dim_pr, dim_qs in dims_2b
        ]

        # Allocated space in MB
        self.allocated_space = (
            8 * (self.data_1b.size + sum([x.size for x in self.data_2b])) / 2**20
        )

    def get_0b_part(self):
        return self.data_0b

    def set_0b_part(self, val):
        self.data_0b = val

    def get_1b_matrix_element(self, ip, iq):
        return self.data_1b[ip, iq]

    def set_1b_matrix_element(self, ip, iq, val):
        self.data_1b[ip, iq] = val

    def get_2b_matrix_element(
        self, local_2bch_index, local_ip, local_iq, local_ir, local_is
    ):
        return self.data_2b[local_2bch_index][local_ip, local_iq, local_ir, local_is]

    def set_2b_matrix_element(
        self, local_2bch_index, local_ip, local_iq, local_ir, local_is, val
    ):
        self.data_2b[local_2bch_index][local_ip, local_iq, local_ir, local_is] = val


class MMHFile:

    @classmethod
    def get_current_version(cls):
        return 1

    @classmethod
    def load_from_binary(cls, fname):
        with open(fname, "rb") as fin:
            f = pickle.load(fin)

        if f.version != MMHFile.get_current_version():
            raise Exception(
                f"MMHFile from {fname} has different version than current version (file: {f.version} vs. curr: {MMHFile.get_current_version()}). Exiting."
            )

        return f

    @classmethod
    def load_from_text(cls, fname):
        return cls(fname)

    def __init__(self, fname):
        self.version = MMHFile.get_current_version()
        self.fs = MMHFileStructure(fname)
        self.fd = MMHFileData(
            len(self.fs.basis),
            [
                (self.fs.ch_dims[ich_pr], self.fs.ch_dims[ich_qs])
                for ich_pr, ich_qs in self.fs.chans_2b
            ],
        )

        # Process 0B part
        self.fd.set_0b_part(_read_0b_part(fname))

        # Process 1B part
        mes_1b = _read_1b_mes(fname)
        for ip, iq in mes_1b:
            self.fd.set_1b_matrix_element(ip, iq, mes_1b[(ip, iq)])

        # Process 2B part
        mes_2b = _read_2b_mes(fname)
        for ip, iq, ir, i_s in mes_2b:
            ich_pr1 = self.fs.state_chans[ip]
            lip = self.fs.state_local_indices[ip]
            ich_pr2 = self.fs.state_chans[ir]
            lir = self.fs.state_local_indices[ir]
            ich_qs1 = self.fs.state_chans[iq]
            liq = self.fs.state_local_indices[iq]
            ich_qs2 = self.fs.state_chans[i_s]
            lis = self.fs.state_local_indices[i_s]
            if ich_pr1 != ich_pr2:
                raise Exception(
                    "Unexpected: p ({}), r ({}) have different channels: {} vs {}".format(
                        ip, ir, ich_pr1, ich_pr2
                    )
                )
            if ich_qs1 != ich_qs2:
                raise Exception(
                    "Unexpected: q ({}), s ({}) have different channels: {} vs {}".format(
                        iq, i_s, ich_qs1, ich_qs2
                    )
                )
            local_2bch_index = self.fs.local_2bch_indices[(ich_pr1, ich_qs1)]

            self.fd.set_2b_matrix_element(
                local_2bch_index, lip, liq, lir, lis, mes_2b[(ip, iq, ir, i_s)]
            )

    def get_0b_part(self):
        return self.fd.get_0b_part()

    def _get_1b_matrix_element_with_internal_index(self, ip, iq):
        return self.fd.get_1b_matrix_element(ip, iq)

    def get_1b_matrix_element(self, p, q):
        ip = self.fs.sid_to_ip_map[p.sid]
        iq = self.fs.sid_to_ip_map[q.sid]
        return self._get_1b_matrix_element_with_internal_index(ip, iq)

    def _get_2b_matrix_element_with_internal_index(self, ip, iq, ir, i_s):
        factor = 1

        # These lookups cost about half the time in this function
        ich_pr1 = self.fs.state_chans[ip]
        lip = self.fs.state_local_indices[ip]
        ich_pr2 = self.fs.state_chans[ir]
        lir = self.fs.state_local_indices[ir]
        ich_qs1 = self.fs.state_chans[iq]
        liq = self.fs.state_local_indices[iq]
        ich_qs2 = self.fs.state_chans[i_s]
        lis = self.fs.state_local_indices[i_s]

        if ich_pr1 != ich_pr2:
            ich_pr1, ich_qs1 = ich_qs1, ich_pr1
            lip, liq = liq, lip
            factor *= -1

        if (ich_pr1 != ich_pr2) or (ich_qs1 != ich_qs2):
            return 0.0

        else:
            if (ich_pr1, ich_qs1) in self.fs.local_2bch_indices:
                local_2bch_index = self.fs.local_2bch_indices[(ich_pr1, ich_qs1)]

                return factor * self.fd.get_2b_matrix_element(
                    local_2bch_index, lip, liq, lir, lis
                )
            else:
                return 0.0

    def get_2b_matrix_element(self, p, q, r, s):
        # These lookups cost about half the time in this function
        ip = self.fs.sid_to_ip_map[p.sid]
        iq = self.fs.sid_to_ip_map[q.sid]
        ir = self.fs.sid_to_ip_map[r.sid]
        i_s = self.fs.sid_to_ip_map[s.sid]
        # ip = self.fs.sid_to_ip_dense_lookup[p.sid]
        # iq = self.fs.sid_to_ip_dense_lookup[q.sid]
        # ir = self.fs.sid_to_ip_dense_lookup[r.sid]
        # i_s = self.fs.sid_to_ip_dense_lookup[s.sid]

        return self._get_2b_matrix_element_with_internal_index(ip, iq, ir, i_s)

    def save_as_text(self, fname):
        PAR_MAP = {-1: 1, 1: 0}
        with open(fname, "w") as fout:
            # Write channels
            fout.write(f"Channels: {len(self.fs.channels)}\n")
            fout.write("i_ch    p   tt   jj |mm|\n")
            for ich, ch in self.fs.channels:
                fout.write(
                    f"{ich:>4} {PAR_MAP[ch.par]:>4} {ch.tt:>4} {ch.jj:>4} {ch.abs_mm:>4}\n"
                )

            # Write states
            fout.write(f"States: {len(self.fs.basis)}\n")
            fout.write(" i_p    n    l   jj   tt   mm\n")
            for ip, p in self.fs.basis:
                fout.write(f"{ip:>4} {p.n:>4} {p.l:>4} {p.jj:>4} {p.tt:>4} {p.mm:>4}\n")

            # Write 0B part
            fout.write(f"0B Part: {self.get_0b_part():>14.6g}\n")

            # Write 1B part
            fout.write("1B Part:\n")
            for ip, _ in self.fs.basis:
                for iq, _ in self.fs.basis:
                    me = self._get_1b_matrix_element_with_internal_index(ip, iq)
                    if abs(me) > 1e-9:
                        fout.write(f"{ip:>4} {iq:>4} {me:>14}\n")

            # Write 2B part
            fout.write("2B Part:\n")
            for ich_pr, ich_qs in self.fs.chans_2b:
                fout.write(f"Chan_pqrs(pr={ich_pr},qs={ich_qs})\n")

                basis_pr = [
                    ip for ip, ich in enumerate(self.fs.state_chans) if ich == ich_pr
                ]
                basis_qs = [
                    iq for iq, ich in enumerate(self.fs.state_chans) if ich == ich_qs
                ]

                for ip in basis_pr:
                    for iq in basis_qs:
                        for ir in basis_pr:
                            for i_s in basis_qs:
                                # This factor just tests that our antisymmetry is working
                                factor = -1
                                me = (
                                    factor
                                    * self._get_2b_matrix_element_with_internal_index(
                                        iq, ip, ir, i_s
                                    )
                                )
                                if abs(me) > 1e-9:
                                    fout.write(
                                        f"{ip:>4} {iq:>4} {ir:>4} {i_s:>4} {me:>14.6g}\n"
                                    )

    def save_as_binary(self, fname, overwrite=False):
        if overwrite:
            print(f"Possibly overwriting existing data at {fname}. This is dangerous.")
        else:
            if pathlib.Path(fname).exists():
                raise Exception(
                    f"File already exists at destination: {fname}. Exiting."
                )

        with open(fname, "wb") as fout:
            pickle.dump(self, fout, protocol=4)

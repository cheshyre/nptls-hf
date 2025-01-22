# Copyright (c) 2024 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from .fast_id import compute_chan_id


def _determine_parity(l):
    # 1 is positive parity, -1 is negative parity
    return 1 - 2 * (l % 2)


class Channel:

    def __init__(self, l, jj, mm, tt):
        self.tt = tt
        self.par = _determine_parity(l)
        self.jj = jj
        self.abs_mm = abs(mm)

        # Unique ID computed based on quantum numbers above
        self.cid = compute_chan_id(self.par, self.jj, self.abs_mm, self.tt)

    def __repr__(self):
        # String representation of channel
        # TODO
        pass

    def __lt__(self, other):
        return self.cid < other.cid

    def __hash__(self):
        return hash(self.cid)

    def __eq__(self, other):
        return self.cid == other.cid

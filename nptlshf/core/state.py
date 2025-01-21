# Copyright (c) 2024 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from .fast_id import compute_state_id
from .chan import Channel


class State:

    def __init__(self, n, l, jj, mm, tt):
        self.n = n
        self.l = l
        self.jj = jj
        self.mm = mm
        self.tt = tt

        # Unique ID computed based on quantum numbers above
        self.sid = compute_state_id(self.n, self.l, self.jj, self.mm, self.tt)

        # Channel and channel ID for state
        self.chan = Channel(self.l, self.jj, self.mm, self.tt)
        self.cid = self.chan.cid

    def __repr__(self):
        return f"| n = {self.n:>2}, l = {self.l:>2}, j = {self.jj:>2}/2, tz = {self.tt:>2}/2, m = {self.mm:>3}/2 >"

    def __lt__(self, other):
        return self.sid < other.sid

    def __hash__(self):
        return hash(self.sid)

    def __eq__(self, other):
        return self.sid == other.sid

# Copyright (c) 2024 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# emax=14, mm values from -29 to 29
SID_WRAP_MM = 60
SID_OFFSET_MM = 30
# emax=14, jj values from 1 to 29
SID_WRAP_JJ = 30
SID_OFFSET_JJ = 0
# emax=14, l values from 0 to 14
SID_WRAP_L = 15
SID_OFFSET_L = 0
# emax=14, n values from 0 to 8
SID_WRAP_N = 10
SID_OFFSET_N = 0
# tt is only ever -1 or 1
SID_WRAP_TT = 2
# max val SID can take (actually 1 beyond max val)
MAX_STATE_ID = SID_WRAP_MM * SID_WRAP_JJ * SID_WRAP_L * SID_WRAP_N * SID_WRAP_TT

# Compute unique ID for state up to emax=14
def compute_state_id(n, l, jj, mm, tt):

    tt_factor = (tt + 1) // 2

    id = (
        tt_factor * SID_WRAP_MM * SID_WRAP_JJ * SID_WRAP_L * SID_WRAP_N
        + (n + SID_OFFSET_N) * SID_WRAP_MM * SID_WRAP_JJ * SID_WRAP_L
        + (l + SID_OFFSET_L) * SID_WRAP_MM * SID_WRAP_JJ
        + (jj + SID_OFFSET_JJ) * SID_WRAP_MM
        + (mm + SID_OFFSET_MM)
    )

    return id


# emax=14, abs_mm values from 1 to 29
CID_WRAP_ABSMM = 30
CID_OFFSET_ABSMM = 0
# emax=14, jj values from 1 to 29
CID_WRAP_JJ = 30
CID_OFFSET_JJ = 0
# parity is only ever -1 or 1
CID_WRAP_PAR = 2
# tt is only ever -1 or 1
CID_WRAP_TT = 2
# max val CID can take (actually 1 beyond max val)
MAX_CHAN_ID = CID_WRAP_ABSMM * CID_WRAP_JJ * CID_WRAP_PAR * CID_WRAP_TT

# Compute unique ID for chan up to emax=14
def compute_chan_id(par, jj, abs_mm, tt):

    tt_factor = (tt + 1) // 2
    par_factor = (par + 1) // 2

    id = (
        tt_factor * CID_WRAP_ABSMM * CID_WRAP_JJ * CID_WRAP_PAR
        + (par_factor) * CID_WRAP_ABSMM * CID_WRAP_JJ
        + (jj + CID_OFFSET_JJ) * CID_WRAP_ABSMM
        + (abs_mm + CID_OFFSET_ABSMM)
    )

    return id

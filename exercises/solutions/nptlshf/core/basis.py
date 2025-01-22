# Copyright (c) 2024 Matthias Heinz
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import re

from .state import State


def _state_sort_key(state):
    # # Sort protons before neutrons
    # tt_val = 1000000 * state.tt
    # Sort states by e = 2 * n + l
    e_val = 10000 * (2 * state.n + state.l)
    # This part is not very clear, but it works...
    # We follow the standard shell model ordering
    kappa = 1
    if state.jj > 2 * state.l:
        kappa = -1
    # Projections are simply ordered according to mm = 2 * m
    # And we list first protons then neutrons
    return e_val + 100 * kappa * state.jj + 2 * state.mm + state.tt


class Basis:

    @classmethod
    def from_emax(cls, emax):
        states = []
        for e in range(emax + 1):
            for l in range(e % 2, e + 1, 2):
                n = (e - l) // 2
                for tt in [-1, 1]:
                    for jj in [2 * l - 1, 2 * l + 1]:
                        if jj < 0:
                            continue
                        for mm in range(-1 * jj, jj + 1, 2):
                            states.append(State(n, l, jj, mm, tt))
        
        states = sorted(states, key=lambda x: _state_sort_key(x))

        return cls(states)


    def __init__(self, states):
        self.states = list(states)
        
        # Automatically computed based on states
        self.channels = sorted(list({state.chan for state in self.states}))

        # Populate states in channels
        self.states_in_channels = []
        for chan in self.channels:
            self.states_in_channels.append([])
            for state_index, state in enumerate(self.states):
                if state.chan == chan:
                    self.states_in_channels[-1].append((state_index, state))

        # Lookup tables generated based on states and channels

    
    def get_states_in_channel(self, chan):
        if chan in self.channels:
            chan_index = self.channels.index(chan)
            return self.states_in_channels[chan_index]
        else:
            return []


    def get_chan_index_and_local_state_index(self, state):
        chan = state.chan
        if chan in self.channels:
            chan_index = self.channels.index(chan)
            if state in self.states_in_channels[chan_index]:
                state_index = self.states_in_channels[chan_index].index(state)
                return chan_index, state_index
            else:
                chan_index, -1
        else:
            return -1, -1
        

def parse_system(system):
    match = re.match(r"(\D+)(\d+)", system)
    elem = match.group(1)
    A = int(match.group(2))
    ELEM = ["n","H","He","Li","Be","B","C","N",
       "O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K",
       "Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y",
       "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
       "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb"]
    Z = ELEM.index(elem)
    N = A - Z
    return A, Z, N
        

def make_occupations(system: str, basis: Basis):
    A, Z, N = parse_system(system)

    dim = len(basis.states)
    occs = [0] * dim

    state_index = 0
    protons_left = Z
    neutrons_left = N
    while (protons_left > 0 or neutrons_left > 0) and (state_index < dim):
        state = basis.states[state_index]

        if state.tt == 1 and protons_left > 0:
            occs[state_index] = 1
            protons_left -= 1

        if state.tt == -1 and neutrons_left > 0:
            occs[state_index] = 1
            neutrons_left -= 1

        state_index += 1

    assert sum(occs) == A

    return occs
